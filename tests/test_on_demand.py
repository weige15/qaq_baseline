import copy
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from qaq.model import install_replacements, prepare_replacements, set_profile
from qaq.on_demand import (convert_nested_model, pack_twos_complement, packed_bytes,
                           reconstruct_packed, unpack_twos_complement)
from qaq.quantization import reconstruct


class PackedPlaneTests(unittest.TestCase):
    def test_all_signed_codes_padding_and_exact_reconstruction(self):
        q = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8).reshape(4, 1, 64)
        scale = torch.tensor([[[0.0]], [[0.125]], [[1.75]], [[0.5]]], dtype=torch.float32)
        planes = pack_twos_complement(q)
        self.assertEqual(planes.dtype, torch.uint8)
        self.assertEqual(planes.shape, (8, packed_bytes(q.numel())))
        self.assertEqual(planes.numel(), q.numel())
        for bits in (4, 6, 8):
            selected = planes[8 - bits:]
            unpacked = unpack_twos_complement(selected, q.numel(), bits).reshape(q.shape)
            shift = 8 - bits
            self.assertTrue(torch.equal(unpacked.to(torch.int16) >> shift,
                                        q.to(torch.int16) >> shift))
            self.assertTrue(torch.equal(
                reconstruct_packed(selected, tuple(q.shape), scale, bits),
                reconstruct(q, scale, bits)))
        self.assertTrue(torch.equal(
            unpack_twos_complement(planes, q.numel(), 8).reshape(q.shape), q))

        odd = q.flatten()[:-1].reshape(3, 1, 85)
        odd_scale = scale[:3]
        padded = pack_twos_complement(odd)
        self.assertEqual(padded.shape, (8, packed_bytes(odd.numel())))
        for plane in padded:
            self.assertEqual(int(plane[-1]) >> 7, 0)
        for bits in (4, 6, 8):
            self.assertTrue(torch.equal(
                reconstruct_packed(padded[8 - bits:], tuple(odd.shape), odd_scale, bits),
                reconstruct(odd, odd_scale, bits)))

    def test_invalid_packed_inputs_are_rejected(self):
        q = torch.tensor([1, -1], dtype=torch.int8)
        planes = pack_twos_complement(q)
        with self.assertRaises(ValueError):
            unpack_twos_complement(planes[4:], 2, 5)
        with self.assertRaises(ValueError):
            unpack_twos_complement(planes[4:, :-1], 2, 4)
        with self.assertRaises(ValueError):
            pack_twos_complement(torch.ones(2))


class TinyQwenStorageTests(unittest.TestCase):
    @staticmethod
    def integrated_model():
        torch.manual_seed(19)
        config = Qwen3Config(hidden_size=128, intermediate_size=256, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=64,
            tie_word_embeddings=True)
        config._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(config).half().eval()
        model.requires_grad_(False)
        install_replacements(prepare_replacements(model))
        return model

    def test_modes_equal_existing_nested_qwen_for_all_precisions_and_mixed_blocks(self):
        reference = self.integrated_model()
        resident_model = copy.deepcopy(reference)
        ondemand_model = copy.deepcopy(reference)
        resident = convert_nested_model(resident_model, "resident")
        ondemand = convert_nested_model(ondemand_model, "ondemand_sync")
        resident.prepare("cpu")
        ondemand.prepare("cpu")
        inputs = torch.tensor([[1, 2, 3, 4]])
        profiles = [[bits] * 4 for bits in (4, 6, 8)] + [[4, 8, 6, 4]]
        try:
            for profile in profiles:
                set_profile(reference, profile)
                resident.set_profile(profile)
                ondemand.set_profile(profile)
                resident.reset_stats()
                ondemand.reset_stats()
                with torch.inference_mode():
                    expected = reference(inputs, use_cache=False).logits
                    actual_resident = resident_model(inputs, use_cache=False).logits
                    actual_ondemand = ondemand_model(inputs, use_cache=False).logits
                self.assertTrue(torch.equal(actual_resident, expected))
                self.assertTrue(torch.equal(actual_ondemand, expected))
                self.assertEqual(resident.stats()["requests"], 4)
                self.assertEqual(resident.stats()["loads"], 0)
                self.assertEqual(resident.stats()["releases"], 0)
                self.assertEqual(resident.stats()["view_closes"], 4)
                self.assertEqual(ondemand.stats()["loads"], 4)
                self.assertEqual(ondemand.stats()["releases"], 4)
                self.assertEqual(ondemand.stats()["active_slots_at_end"], 0)
                self.assertEqual(ondemand.stats()["max_active_slots"], 1)
                expected_copied = sum(
                    profile[i] * payload.planes.shape[1] + payload.scale_bytes
                    for i, payload in enumerate(ondemand.payloads))
                self.assertEqual(ondemand.stats()["copied_bytes"], expected_copied)
                self.assertEqual(ondemand.stats()["requested_bytes"], expected_copied)
            self.assertEqual(resident.inventory()["source_devices"], ["cpu"])
            self.assertEqual(ondemand.inventory()["source_devices"], ["cpu"])
            self.assertGreater(ondemand.inventory()["slot_capacity_bytes"], 0)
        finally:
            resident.close()
            ondemand.close()

    def test_slot_is_released_when_a_projection_raises(self):
        model = self.integrated_model()
        storage = convert_nested_model(model, "ondemand_sync")
        storage.prepare("cpu")

        def fail(_module, _args):
            raise RuntimeError("injected")

        handle = model.model.layers[0].self_attn.q_proj.register_forward_pre_hook(fail)
        try:
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with torch.inference_mode():
                    model(torch.tensor([[1, 2]]), use_cache=False)
            self.assertEqual(storage.stats()["loads"], 1)
            self.assertEqual(storage.stats()["releases"], 1)
            self.assertEqual(storage.stats()["active_slots_at_end"], 0)
            self.assertTrue(storage.events[0]["released"])
        finally:
            handle.remove()
            storage.close()


if __name__ == "__main__":
    unittest.main()
