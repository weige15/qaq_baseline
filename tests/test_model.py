import tempfile
from pathlib import Path
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from qaq.model import (NestedLinear, block_parameter_counts, blocks, install_replacements,
                       load_quantized, prepare_replacements, save_quantized, set_profile)
from qaq.quantization import apply_fixed, reconstruct


class IntegratedModelTests(unittest.TestCase):
    def test_independent_blocks_exact_reference_and_checkpoint(self):
        torch.manual_seed(4)
        config = Qwen3Config(hidden_size=128, intermediate_size=256, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=64,
            tie_word_embeddings=True)
        config._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(config).half().eval()
        model.requires_grad_(False)
        replacements = prepare_replacements(model)
        apply_fixed(model, 8)
        for parent, name, packed, _ in replacements:
            self.assertTrue(torch.equal(getattr(parent, name).weight,
                                        reconstruct(packed.q, packed.scale, 8)))
        x = torch.tensor([[1,2,3,4]])
        with torch.no_grad():
            reference = model(x, use_cache=False).logits
        install_replacements(replacements)
        with torch.no_grad():
            full8 = model(x, use_cache=False).logits
        self.assertTrue(torch.equal(full8, reference))
        before = [b.q_proj.active_weight.clone() if i%2==0 else b.gate_proj.active_weight.clone()
                  for i,b in enumerate(blocks(model))]
        set_profile(model, [4,8,8,8])
        with torch.no_grad():
            attn4 = model(x, use_cache=False).logits
        self.assertFalse(torch.equal(attn4, full8))
        self.assertFalse(torch.equal(model.model.layers[0].self_attn.q_proj.active_weight, before[0]))
        self.assertTrue(torch.equal(model.model.layers[0].mlp.gate_proj.active_weight, before[1]))
        set_profile(model, [8,4,8,8])
        with torch.no_grad():
            ffn4 = model(x, use_cache=False).logits
        self.assertFalse(torch.equal(ffn4, full8))
        self.assertTrue(torch.equal(model.model.layers[0].self_attn.q_proj.active_weight, before[0]))
        self.assertFalse(torch.equal(model.model.layers[0].mlp.gate_proj.active_weight, before[1]))
        counts = block_parameter_counts(model)
        self.assertEqual(counts[0], counts[2])
        self.assertEqual(counts[1], counts[3])
        with self.assertRaises(ValueError):
            set_profile(model, [8,8,8,5])
        self.assertEqual(model.model.layers[0].mlp.gate_proj.bits, 4)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"one-model.pt"
            save_quantized(model, path, {"group_size": 128})
            stored = torch.load(path, weights_only=True)
            self.assertFalse(any("active_weight" in k for k in stored["state_dict"]))
            self.assertEqual(sum(k.endswith(".q") for k in stored["state_dict"]),14)
            loaded, _ = load_quantized(path)
            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, loaded.state_dict()[name]), name)
            for bits in (4,6,8):
                set_profile(model, [bits]*4)
                set_profile(loaded, [bits]*4)
                with torch.no_grad():
                    self.assertTrue(torch.equal(model(x, use_cache=False).logits,
                                                loaded(x, use_cache=False).logits))
            self.assertTrue(all(isinstance(b.q_proj, NestedLinear) for i,b in enumerate(blocks(loaded)) if i%2==0))


if __name__ == "__main__":
    unittest.main()
