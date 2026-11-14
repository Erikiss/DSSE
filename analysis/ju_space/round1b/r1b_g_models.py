"""Pinned Gemma model manifests and a metadata-only audit (part G, phase 1).

No torch/transformers imports and no weight downloads. The LFS SHA-256 of a
weight or tokenizer artifact comes from HF metadata, not a downloaded blob.
`refresh` writes a reviewable JSON report outside the repository; it never
updates the checked-in manifest or grants permission for a GPU run.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
import urllib.parse
import urllib.request

AUDITED_AT = "2026-11-14"
MODEL_REPOS = {
    "gemma4-e2b": "google/gemma-4-E2B-it",
    "gemma4-e4b": "google/gemma-4-E4B-it",
    "gemma4-31b-qat": "google/gemma-4-31B-it-qat-w4a16-ct",
    "gemma4-12b-qat": "google/gemma-4-12B-it-qat-w4a16-ct",
    "gemma4-e2b-qat": "google/gemma-4-E2B-it-qat-w4a16-ct",
    "gemma4-e4b-qat": "google/gemma-4-E4B-it-qat-w4a16-ct",
    "gemma4-e2b-assistant": "google/gemma-4-E2B-it-assistant",
    "gemma4-e4b-assistant": "google/gemma-4-E4B-it-assistant",
}
# Metadata audit result. Every remote source below is pinned to its commit.
MODEL_SPECS = {'gemma4-e2b': {'repo': 'google/gemma-4-E2B-it',
                'revision': '4adf15748de9e1a0097b6274cee9c15836c41c37',
                'family': 'gemma4',
                'architecture': 'Gemma4ForConditionalGeneration',
                'role': 'target',
                'target': None,
                'layers': 35,
                'hidden': 1536,
                'widths': [6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144,
                           12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288,
                           12288, 12288, 12288, 12288, 12288, 12288, 12288],
                'layer_types': ['sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention',
                                'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention',
                                'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention',
                                'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention',
                                'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention',
                                'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention',
                                'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention',
                                'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention',
                                'sliding_attention', 'sliding_attention', 'full_attention'],
                'sliding_window': 512,
                'max_position_embeddings': 131072,
                'vocab_size': 262144,
                'num_kv_shared_layers': 20,
                'hidden_size_per_layer_input': 256,
                'final_logit_softcapping': 30.0,
                'dtype': 'float32',
                'quantized': False,
                'forward_only': False,
                'gradient_measurements': True,
                'license': 'apache-2.0',
                'gated': False,
                'license_url': 'https://ai.google.dev/gemma/docs/gemma_4_license',
                'parameters': 5123178051,
                'stored_tensor_metadata': {'parameters': {'BF16': 5123178051}, 'total': 5123178051},
                'weight_bytes': 10246621918,
                'sha256': {'chat_template.jinja': '0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5',
                           'config.json': '1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330',
                           'generation_config.json': 'd4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de',
                           'model.safetensors': '2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550',
                           'processor_config.json': '32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c',
                           'tokenizer.json': 'ab26b2add729e6f645bd869f228f14190a06615538f3be6f7786f33d13d78ad2',
                           'tokenizer_config.json': '9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633'},
                'file_bytes': {'chat_template.jinja': 18569,
                               'config.json': 4954,
                               'generation_config.json': 208,
                               'model.safetensors': 10246621918,
                               'processor_config.json': 1689,
                               'tokenizer.json': 32155733,
                               'tokenizer_config.json': 3082},
                'checksum_evidence': {'chat_template.jinja': 'SHA-256 of fetched small file',
                                      'config.json': 'SHA-256 of fetched small file',
                                      'generation_config.json': 'SHA-256 of fetched small file',
                                      'model.safetensors': 'HF LFS SHA-256 metadata (body not downloaded)',
                                      'processor_config.json': 'SHA-256 of fetched small file',
                                      'tokenizer.json': 'HF LFS SHA-256 metadata (body not downloaded)',
                                      'tokenizer_config.json': 'SHA-256 of fetched small file'},
                'card_sha256': '3e0608a6be80e3eb040a5aa6c76707809503deb85be6360184a118fb6156526c',
                'sources': {'metadata': 'https://huggingface.co/api/models/google/gemma-4-E2B-it/revision/4adf15748de9e1a0097b6274cee9c15836c41c37?blobs=true',
                            'config': 'https://huggingface.co/google/gemma-4-E2B-it/resolve/4adf15748de9e1a0097b6274cee9c15836c41c37/config.json',
                            'card': 'https://huggingface.co/google/gemma-4-E2B-it/blob/4adf15748de9e1a0097b6274cee9c15836c41c37/README.md'}},
 'gemma4-e4b': {'repo': 'google/gemma-4-E4B-it',
                'revision': '4ecc2ba63bcac23ba4a767fc87025b1450f84c9c',
                'family': 'gemma4',
                'architecture': 'Gemma4ForConditionalGeneration',
                'role': 'target',
                'target': None,
                'layers': 42,
                'hidden': 2560,
                'widths': [10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240,
                           10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240,
                           10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240,
                           10240, 10240, 10240],
                'layer_types': ['sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention',
                                'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention',
                                'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention',
                                'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention',
                                'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention',
                                'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention',
                                'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention',
                                'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention',
                                'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention',
                                'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention',
                                'sliding_attention', 'full_attention'],
                'sliding_window': 512,
                'max_position_embeddings': 131072,
                'vocab_size': 262144,
                'num_kv_shared_layers': 18,
                'hidden_size_per_layer_input': 256,
                'final_logit_softcapping': 30.0,
                'dtype': 'float32',
                'quantized': False,
                'forward_only': False,
                'gradient_measurements': True,
                'license': 'apache-2.0',
                'gated': False,
                'license_url': 'https://ai.google.dev/gemma/docs/gemma_4_license',
                'parameters': 7996156490,
                'stored_tensor_metadata': {'parameters': {'BF16': 7996156490}, 'total': 7996156490},
                'weight_bytes': 15992595884,
                'sha256': {'chat_template.jinja': '0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5',
                           'config.json': '33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4',
                           'generation_config.json': 'd4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de',
                           'model.safetensors': 'cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503',
                           'processor_config.json': '32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c',
                           'tokenizer.json': '91d9cf86906f78e197021fda88214d448e5c120665d0243a8e776e046986c6bc',
                           'tokenizer_config.json': '9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633'},
                'file_bytes': {'chat_template.jinja': 18569,
                               'config.json': 5145,
                               'generation_config.json': 208,
                               'model.safetensors': 15992595884,
                               'processor_config.json': 1689,
                               'tokenizer.json': 32156428,
                               'tokenizer_config.json': 3082},
                'checksum_evidence': {'chat_template.jinja': 'SHA-256 of fetched small file',
                                      'config.json': 'SHA-256 of fetched small file',
                                      'generation_config.json': 'SHA-256 of fetched small file',
                                      'model.safetensors': 'HF LFS SHA-256 metadata (body not downloaded)',
                                      'processor_config.json': 'SHA-256 of fetched small file',
                                      'tokenizer.json': 'HF LFS SHA-256 metadata (body not downloaded)',
                                      'tokenizer_config.json': 'SHA-256 of fetched small file'},
                'card_sha256': 'b21e4f69614ccd77baa2f3797d05311040dee07b989cb9f0d25111aa4b605b2c',
                'sources': {'metadata': 'https://huggingface.co/api/models/google/gemma-4-E4B-it/revision/4ecc2ba63bcac23ba4a767fc87025b1450f84c9c?blobs=true',
                            'config': 'https://huggingface.co/google/gemma-4-E4B-it/resolve/4ecc2ba63bcac23ba4a767fc87025b1450f84c9c/config.json',
                            'card': 'https://huggingface.co/google/gemma-4-E4B-it/blob/4ecc2ba63bcac23ba4a767fc87025b1450f84c9c/README.md'}},
 'gemma4-31b-qat': {'repo': 'google/gemma-4-31B-it-qat-w4a16-ct',
                    'revision': '52f3f65bc7a02d555763bc923bd1d9094898219d',
                    'family': 'gemma4',
                    'architecture': 'Gemma4ForConditionalGeneration',
                    'role': 'target',
                    'target': None,
                    'layers': 60,
                    'hidden': 5376,
                    'widths': [21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504,
                               21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504,
                               21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504,
                               21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504,
                               21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504, 21504],
                    'layer_types': ['sliding_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'full_attention'],
                    'sliding_window': 1024,
                    'max_position_embeddings': 262144,
                    'vocab_size': 262144,
                    'num_kv_shared_layers': 0,
                    'hidden_size_per_layer_input': 0,
                    'final_logit_softcapping': 30.0,
                    'dtype': 'float16',
                    'quantized': True,
                    'forward_only': True,
                    'gradient_measurements': False,
                    'license': 'apache-2.0',
                    'gated': False,
                    'license_url': 'https://ai.google.dev/gemma/docs/gemma_4_license',
                    'parameters': None,
                    'stored_tensor_metadata': {'parameters': {'I32': 29286727680, 'BF16': 3395647340},
                                               'total': 32682375020},
                    'weight_bytes': 23265352448,
                    'sha256': {'chat_template.jinja': 'ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4',
                               'config.json': 'b100d85e571c25b688e82919d819253063a1defb0e6cee3b9b1c1fbad99bd0c2',
                               'generation_config.json': '70110aa886fa7ee25ce47346d102dd29ca5f967500490b4362fc72068e1db70a',
                               'model.safetensors': '1b9b1d622a93f02c0d33f98e502f233b5d707443af6ddc464ed0bf5498506c20',
                               'processor_config.json': '32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c',
                               'tokenizer.json': 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f',
                               'tokenizer_config.json': 'b8045a4576903e86903291d5cbdd4adfc8859e9ce3c98621bdbd957f73ed394b'},
                    'file_bytes': {'chat_template.jinja': 18683,
                                   'config.json': 18711,
                                   'generation_config.json': 208,
                                   'model.safetensors': 23265352448,
                                   'processor_config.json': 1689,
                                   'tokenizer.json': 32169626,
                                   'tokenizer_config.json': 3728},
                    'checksum_evidence': {'chat_template.jinja': 'SHA-256 of fetched small file',
                                          'config.json': 'SHA-256 of fetched small file',
                                          'generation_config.json': 'SHA-256 of fetched small file',
                                          'model.safetensors': 'HF LFS SHA-256 metadata (body not downloaded)',
                                          'processor_config.json': 'SHA-256 of fetched small file',
                                          'tokenizer.json': 'HF LFS SHA-256 metadata (body not downloaded)',
                                          'tokenizer_config.json': 'SHA-256 of fetched small file'},
                    'card_sha256': '46bc522c1aaef71174fd93b842cbd86b6d4101bd92963fba31ca48cdf0c3d787',
                    'sources': {'metadata': 'https://huggingface.co/api/models/google/gemma-4-31B-it-qat-w4a16-ct/revision/52f3f65bc7a02d555763bc923bd1d9094898219d?blobs=true',
                                'config': 'https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct/resolve/52f3f65bc7a02d555763bc923bd1d9094898219d/config.json',
                                'card': 'https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct/blob/52f3f65bc7a02d555763bc923bd1d9094898219d/README.md'}},
 'gemma4-12b-qat': {'repo': 'google/gemma-4-12B-it-qat-w4a16-ct',
                    'revision': '1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee',
                    'family': 'gemma4_unified',
                    'architecture': 'Gemma4UnifiedForConditionalGeneration',
                    'role': 'target',
                    'target': None,
                    'layers': 48,
                    'hidden': 3840,
                    'widths': [15360, 15360, 15360, 15360, 15360, 15360, 15360, 15360, 15360, 15360, 15360, 15360,
                               15360, 15360, 15360, 15360, 15360, 15360, 15360, 15360, 15360, 15360, 15360, 15360,
                               15360, 15360, 15360, 15360, 15360, 15360, 15360, 15360, 15360, 15360, 15360, 15360,
                               15360, 15360, 15360, 15360, 15360, 15360, 15360, 15360, 15360, 15360, 15360, 15360],
                    'layer_types': ['sliding_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'full_attention'],
                    'sliding_window': 1024,
                    'max_position_embeddings': 262144,
                    'vocab_size': 262144,
                    'num_kv_shared_layers': 0,
                    'hidden_size_per_layer_input': 0,
                    'final_logit_softcapping': 30.0,
                    'dtype': 'float16',
                    'quantized': True,
                    'forward_only': True,
                    'gradient_measurements': False,
                    'license': 'apache-2.0',
                    'gated': False,
                    'license_url': 'https://ai.google.dev/gemma/docs/gemma_4_license',
                    'parameters': None,
                    'stored_tensor_metadata': {'parameters': {'I32': 10899947520, 'BF16': 2066415664},
                                               'total': 12966363184},
                    'weight_bytes': 10264229896,
                    'sha256': {'chat_template.jinja': 'ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4',
                               'config.json': 'aa3ba82096a857e8b9a157df21a50df15a67e889a84b7580a4eb4f2b86c28d78',
                               'generation_config.json': 'c70f87dc2995fc43406c0bcfb41b69c6d31c2d0c033fa09e536ffabc091ae24c',
                               'model.safetensors': '60b6e3989502969d8ae04185d72ecbbc7db63978d5af747a493d53895aa6bfa3',
                               'processor_config.json': '6b938e76555b3e9946890770e1abcd442a4718f34041a58e8139dc8ad34545c9',
                               'tokenizer.json': 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f',
                               'tokenizer_config.json': '13c6e1a5abadc12374393fa5fa352682f7326d8091b60cc66d9cb90e640da9b9'},
                    'file_bytes': {'chat_template.jinja': 18683,
                                   'config.json': 6183,
                                   'generation_config.json': 255,
                                   'model.safetensors': 10264229896,
                                   'processor_config.json': 1382,
                                   'tokenizer.json': 32169626,
                                   'tokenizer_config.json': 3761},
                    'checksum_evidence': {'chat_template.jinja': 'SHA-256 of fetched small file',
                                          'config.json': 'SHA-256 of fetched small file',
                                          'generation_config.json': 'SHA-256 of fetched small file',
                                          'model.safetensors': 'HF LFS SHA-256 metadata (body not downloaded)',
                                          'processor_config.json': 'SHA-256 of fetched small file',
                                          'tokenizer.json': 'HF LFS SHA-256 metadata (body not downloaded)',
                                          'tokenizer_config.json': 'SHA-256 of fetched small file'},
                    'card_sha256': '68c5824b37ace62bab0bc556ab593cd6a949f1e6d4b80eb9ceb27ca85cbf6cd2',
                    'sources': {'metadata': 'https://huggingface.co/api/models/google/gemma-4-12B-it-qat-w4a16-ct/revision/1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee?blobs=true',
                                'config': 'https://huggingface.co/google/gemma-4-12B-it-qat-w4a16-ct/resolve/1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee/config.json',
                                'card': 'https://huggingface.co/google/gemma-4-12B-it-qat-w4a16-ct/blob/1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee/README.md'}},
 'gemma4-e2b-qat': {'repo': 'google/gemma-4-E2B-it-qat-w4a16-ct',
                    'revision': '971342c08f607aa7779983f6b5289778b5d271a7',
                    'family': 'gemma4',
                    'architecture': 'Gemma4ForConditionalGeneration',
                    'role': 'target',
                    'target': None,
                    'layers': 35,
                    'hidden': 1536,
                    'widths': [6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144, 6144,
                               6144, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288,
                               12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288, 12288],
                    'layer_types': ['sliding_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'full_attention'],
                    'sliding_window': 512,
                    'max_position_embeddings': 131072,
                    'vocab_size': 262144,
                    'num_kv_shared_layers': 20,
                    'hidden_size_per_layer_input': 256,
                    'final_logit_softcapping': 30.0,
                    'dtype': 'float16',
                    'quantized': True,
                    'forward_only': True,
                    'gradient_measurements': False,
                    'license': 'apache-2.0',
                    'gated': False,
                    'license_url': 'https://ai.google.dev/gemma/docs/gemma_4_license',
                    'parameters': None,
                    'stored_tensor_metadata': {'parameters': {'I32': 1876819968, 'BF16': 3630130755},
                                               'total': 5506950723},
                    'weight_bytes': 8316306646,
                    'sha256': {'chat_template.jinja': '0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5',
                               'config.json': 'd1b2018bd9bd9459ce40b2998c0e32ab7df693053520cfbcdcf47a93cd4410d7',
                               'generation_config.json': '70110aa886fa7ee25ce47346d102dd29ca5f967500490b4362fc72068e1db70a',
                               'model.safetensors': '93177bfc1b53823f2e01c7cebc3b94f65d189e373a647d086112f614ba448ab9',
                               'processor_config.json': '32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c',
                               'tokenizer.json': 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f',
                               'tokenizer_config.json': 'b8045a4576903e86903291d5cbdd4adfc8859e9ce3c98621bdbd957f73ed394b'},
                    'file_bytes': {'chat_template.jinja': 18569,
                                   'config.json': 22202,
                                   'generation_config.json': 208,
                                   'model.safetensors': 8316306646,
                                   'processor_config.json': 1689,
                                   'tokenizer.json': 32169626,
                                   'tokenizer_config.json': 3728},
                    'checksum_evidence': {'chat_template.jinja': 'SHA-256 of fetched small file',
                                          'config.json': 'SHA-256 of fetched small file',
                                          'generation_config.json': 'SHA-256 of fetched small file',
                                          'model.safetensors': 'HF LFS SHA-256 metadata (body not downloaded)',
                                          'processor_config.json': 'SHA-256 of fetched small file',
                                          'tokenizer.json': 'HF LFS SHA-256 metadata (body not downloaded)',
                                          'tokenizer_config.json': 'SHA-256 of fetched small file'},
                    'card_sha256': '2e67a9e7676a850b8b875ecec1d8309d7f4cf15fc126f45faeb914305bc77e76',
                    'sources': {'metadata': 'https://huggingface.co/api/models/google/gemma-4-E2B-it-qat-w4a16-ct/revision/971342c08f607aa7779983f6b5289778b5d271a7?blobs=true',
                                'config': 'https://huggingface.co/google/gemma-4-E2B-it-qat-w4a16-ct/resolve/971342c08f607aa7779983f6b5289778b5d271a7/config.json',
                                'card': 'https://huggingface.co/google/gemma-4-E2B-it-qat-w4a16-ct/blob/971342c08f607aa7779983f6b5289778b5d271a7/README.md'}},
 'gemma4-e4b-qat': {'repo': 'google/gemma-4-E4B-it-qat-w4a16-ct',
                    'revision': '6cd26aaa2357fb2bad8c51699a7558a4d1a965bb',
                    'family': 'gemma4',
                    'architecture': 'Gemma4ForConditionalGeneration',
                    'role': 'target',
                    'target': None,
                    'layers': 42,
                    'hidden': 2560,
                    'widths': [10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240,
                               10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240,
                               10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240, 10240,
                               10240, 10240, 10240, 10240, 10240, 10240],
                    'layer_types': ['sliding_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention',
                                    'sliding_attention', 'sliding_attention', 'sliding_attention',
                                    'sliding_attention', 'sliding_attention', 'full_attention'],
                    'sliding_window': 512,
                    'max_position_embeddings': 131072,
                    'vocab_size': 262144,
                    'num_kv_shared_layers': 18,
                    'hidden_size_per_layer_input': 256,
                    'final_logit_softcapping': 30.0,
                    'dtype': 'float16',
                    'quantized': True,
                    'forward_only': True,
                    'gradient_measurements': False,
                    'license': 'apache-2.0',
                    'gated': False,
                    'license_url': 'https://ai.google.dev/gemma/docs/gemma_4_license',
                    'parameters': None,
                    'stored_tensor_metadata': {'parameters': {'I32': 3972792320, 'BF16': 4639397194},
                                               'total': 8612189514},
                    'weight_bytes': 11513861828,
                    'sha256': {'chat_template.jinja': '0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5',
                               'config.json': '7f0cd2d17bdb0d5adc6db8e7168bb9a9aa68a6c4ad879d80ecf83e6819c893d3',
                               'generation_config.json': '70110aa886fa7ee25ce47346d102dd29ca5f967500490b4362fc72068e1db70a',
                               'model.safetensors': '19d89d2a4b3e401a3eae75ec08ab112810421a036c91d8c8226b686be7fe0c5d',
                               'processor_config.json': '32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c',
                               'tokenizer.json': 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f',
                               'tokenizer_config.json': 'b8045a4576903e86903291d5cbdd4adfc8859e9ce3c98621bdbd957f73ed394b'},
                    'file_bytes': {'chat_template.jinja': 18569,
                                   'config.json': 22393,
                                   'generation_config.json': 208,
                                   'model.safetensors': 11513861828,
                                   'processor_config.json': 1689,
                                   'tokenizer.json': 32169626,
                                   'tokenizer_config.json': 3728},
                    'checksum_evidence': {'chat_template.jinja': 'SHA-256 of fetched small file',
                                          'config.json': 'SHA-256 of fetched small file',
                                          'generation_config.json': 'SHA-256 of fetched small file',
                                          'model.safetensors': 'HF LFS SHA-256 metadata (body not downloaded)',
                                          'processor_config.json': 'SHA-256 of fetched small file',
                                          'tokenizer.json': 'HF LFS SHA-256 metadata (body not downloaded)',
                                          'tokenizer_config.json': 'SHA-256 of fetched small file'},
                    'card_sha256': '027d8519c86f73bdd46a08803f4866a9b51783840cdaf3384077297fea169f6f',
                    'sources': {'metadata': 'https://huggingface.co/api/models/google/gemma-4-E4B-it-qat-w4a16-ct/revision/6cd26aaa2357fb2bad8c51699a7558a4d1a965bb?blobs=true',
                                'config': 'https://huggingface.co/google/gemma-4-E4B-it-qat-w4a16-ct/resolve/6cd26aaa2357fb2bad8c51699a7558a4d1a965bb/config.json',
                                'card': 'https://huggingface.co/google/gemma-4-E4B-it-qat-w4a16-ct/blob/6cd26aaa2357fb2bad8c51699a7558a4d1a965bb/README.md'}},
 'gemma4-e2b-assistant': {'repo': 'google/gemma-4-E2B-it-assistant',
                          'revision': '2d874ef7d29f9a30599a1e4b3c1cbc9595f005df',
                          'family': 'gemma4_assistant',
                          'architecture': 'Gemma4AssistantForCausalLM',
                          'role': 'assistant',
                          'target': 'gemma4-e2b',
                          'layers': 4,
                          'hidden': 256,
                          'widths': [2048, 2048, 2048, 2048],
                          'layer_types': ['sliding_attention', 'sliding_attention', 'sliding_attention',
                                          'full_attention'],
                          'sliding_window': 512,
                          'max_position_embeddings': 131072,
                          'vocab_size': 262144,
                          'num_kv_shared_layers': 4,
                          'hidden_size_per_layer_input': 0,
                          'final_logit_softcapping': None,
                          'dtype': 'float32',
                          'quantized': False,
                          'forward_only': False,
                          'gradient_measurements': False,
                          'license': 'apache-2.0',
                          'gated': False,
                          'license_url': 'https://ai.google.dev/gemma/docs/gemma_4_license',
                          'parameters': 77993476,
                          'stored_tensor_metadata': {'parameters': {'I64': 262144, 'BF16': 77731332},
                                                     'total': 77993476},
                          'weight_bytes': 157565344,
                          'sha256': {'config.json': '7f42f559a6a69ffaeaf6b61a1ece3a562a2ed5ad00b8d30f16917ba5ab1bcbe9',
                                     'generation_config.json': '8e58004dc0e2407b63410b190bb8470efbdcfeb71533f1770e09c20abe193a6f',
                                     'model.safetensors': '93682eb1c97639d18f007704dc880bd74cbe530adaf7b1bb561213863fdad2a6',
                                     'tokenizer.json': '75a6583c1a418e2bbd79c60d95d28e0f5bf549ad3f2990b5bdb5238c6c2bf70c',
                                     'tokenizer_config.json': '089594a3924fcfd4cb1c596a7906fbf476193519e5198f780912eed02b177e42'},
                          'file_bytes': {'config.json': 2315,
                                         'generation_config.json': 307,
                                         'model.safetensors': 157565344,
                                         'tokenizer.json': 32169440,
                                         'tokenizer_config.json': 822},
                          'checksum_evidence': {'config.json': 'SHA-256 of fetched small file',
                                                'generation_config.json': 'SHA-256 of fetched small file',
                                                'model.safetensors': 'HF LFS SHA-256 metadata (body not downloaded)',
                                                'tokenizer.json': 'HF LFS SHA-256 metadata (body not downloaded)',
                                                'tokenizer_config.json': 'SHA-256 of fetched small file'},
                          'card_sha256': '6d6e15f44732a1984951b0d6e258f84b0e97f3399707aada2c9b0b6c0b4ca3eb',
                          'sources': {'metadata': 'https://huggingface.co/api/models/google/gemma-4-E2B-it-assistant/revision/2d874ef7d29f9a30599a1e4b3c1cbc9595f005df?blobs=true',
                                      'config': 'https://huggingface.co/google/gemma-4-E2B-it-assistant/resolve/2d874ef7d29f9a30599a1e4b3c1cbc9595f005df/config.json',
                                      'card': 'https://huggingface.co/google/gemma-4-E2B-it-assistant/blob/2d874ef7d29f9a30599a1e4b3c1cbc9595f005df/README.md'}},
 'gemma4-e4b-assistant': {'repo': 'google/gemma-4-E4B-it-assistant',
                          'revision': '8d0031ea8c2109e2b1e86bb9368a4539b537f80a',
                          'family': 'gemma4_assistant',
                          'architecture': 'Gemma4AssistantForCausalLM',
                          'role': 'assistant',
                          'target': 'gemma4-e4b',
                          'layers': 4,
                          'hidden': 256,
                          'widths': [2048, 2048, 2048, 2048],
                          'layer_types': ['sliding_attention', 'sliding_attention', 'sliding_attention',
                                          'full_attention'],
                          'sliding_window': 512,
                          'max_position_embeddings': 131072,
                          'vocab_size': 262144,
                          'num_kv_shared_layers': 4,
                          'hidden_size_per_layer_input': 0,
                          'final_logit_softcapping': None,
                          'dtype': 'float32',
                          'quantized': False,
                          'forward_only': False,
                          'gradient_measurements': False,
                          'license': 'apache-2.0',
                          'gated': False,
                          'license_url': 'https://ai.google.dev/gemma/docs/gemma_4_license',
                          'parameters': 78779908,
                          'stored_tensor_metadata': {'parameters': {'I64': 262144, 'BF16': 78517764},
                                                     'total': 78779908},
                          'weight_bytes': 159138208,
                          'sha256': {'config.json': '13cdf5418e5a89cf951717a953cd11cef4e5c2743c4767e9e5dd72802954f822',
                                     'generation_config.json': '8e58004dc0e2407b63410b190bb8470efbdcfeb71533f1770e09c20abe193a6f',
                                     'model.safetensors': '12875062fc25c51e8fa9b62abd2de7ad48b7d63f8559d5d604fbd5a3d6bcff16',
                                     'tokenizer.json': '75a6583c1a418e2bbd79c60d95d28e0f5bf549ad3f2990b5bdb5238c6c2bf70c',
                                     'tokenizer_config.json': '089594a3924fcfd4cb1c596a7906fbf476193519e5198f780912eed02b177e42'},
                          'file_bytes': {'config.json': 2315,
                                         'generation_config.json': 307,
                                         'model.safetensors': 159138208,
                                         'tokenizer.json': 32169440,
                                         'tokenizer_config.json': 822},
                          'checksum_evidence': {'config.json': 'SHA-256 of fetched small file',
                                                'generation_config.json': 'SHA-256 of fetched small file',
                                                'model.safetensors': 'HF LFS SHA-256 metadata (body not downloaded)',
                                                'tokenizer.json': 'HF LFS SHA-256 metadata (body not downloaded)',
                                                'tokenizer_config.json': 'SHA-256 of fetched small file'},
                          'card_sha256': '65e7bdd8dcd485494401fdf3c0b6d818cb086d6ef8f69958bae1ddc4358197f1',
                          'sources': {'metadata': 'https://huggingface.co/api/models/google/gemma-4-E4B-it-assistant/revision/8d0031ea8c2109e2b1e86bb9368a4539b537f80a?blobs=true',
                                      'config': 'https://huggingface.co/google/gemma-4-E4B-it-assistant/resolve/8d0031ea8c2109e2b1e86bb9368a4539b537f80a/config.json',
                                      'card': 'https://huggingface.co/google/gemma-4-E4B-it-assistant/blob/8d0031ea8c2109e2b1e86bb9368a4539b537f80a/README.md'}}}

LENS_SOURCES = {
    "neuron_lens": {
        "repo": "stdlogits/neuron-lens", "repo_type": "dataset",
        "revision": "21666b9f217328caee73d10de85ca19f4c174e72",
        "license": "cc-by-4.0", "base_model": "google/gemma-4-12B",
        "coordinates": ["layer", "neuron", "pole"],
        "index_base": 0, "layers": 48, "neurons_per_layer": 15360,
        "poles": [-1, 1], "target_equivalence": "unvalidated: base versus IT-QAT",
        "url": "https://huggingface.co/datasets/stdlogits/neuron-lens",
    },
    "rs_lenses": {
        "repo": "stdlogits/rs-lenses", "repo_type": "model",
        "revision": "ee42b62072dd04f375c110990678fc5176c3ed4c",
        "license": "cc-by-4.0", "base_model": "google/gemma-4-12B",
        "r_lens_path": "gemma-4-12b/r-lens/lens.pt",
        "url": "https://huggingface.co/stdlogits/rs-lenses",
    },
}


def get_spec(name: str) -> dict:
    """Return an independent manifest for a registered alias or exact HF ID."""
    if name not in MODEL_SPECS:
        aliases = [a for a, s in MODEL_SPECS.items() if s["repo"] == name]
        if not aliases:
            raise KeyError(f"unknown Gemma model {name!r}")
        name = aliases[0]
    return copy.deepcopy(MODEL_SPECS[name])


# A small compatibility spelling for callers; both return independent data.
spec = get_spec


def _fetch_small(url: str, limit: int = 2 * 1024 * 1024) -> bytes:
    """Read only HF metadata/text. Weight/blob requests cannot use this helper."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co":
        raise ValueError("audit permits only HTTPS huggingface.co metadata")
    if parsed.path.endswith((".safetensors", ".bin", ".pt", ".pth", ".gguf", ".model")):
        raise ValueError("metadata audit must not download weight/tokenizer binaries")
    req = urllib.request.Request(url, headers={"User-Agent": "DSSE-part-G-metadata-audit/1"})
    with urllib.request.urlopen(req, timeout=30) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"metadata resource exceeds {limit} bytes")
    return data


def _safe_path(name: str) -> bool:
    p = PurePosixPath(name)
    return bool(name and not p.is_absolute() and ".." not in p.parts and "\\" not in name)


def _artifact(name: str) -> bool:
    return (name.endswith((".json", ".jinja", ".safetensors", ".model", ".txt"))
            or name in ("merges.txt", "vocab.json")) and not name.startswith(".")


def audit_model(name: str, revision: str | None = None) -> dict:
    """Fetch one manifest. Non-LFS small files are hashed, LFS bodies never read.

    `revision=None` inspects current main but records its immutable SHA before
    requesting files; a pinned revision permits reproducible verification.
    """
    if name not in MODEL_REPOS:
        raise KeyError(name)
    repo = MODEL_REPOS[name]
    api = f"https://huggingface.co/api/models/{repo}"
    if revision:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("revision must be an immutable 40-character commit")
        api += f"/revision/{revision}"
    metadata = json.loads(_fetch_small(api + "?blobs=true"))
    rev = metadata["sha"]
    if revision and rev != revision:
        raise ValueError("HF returned a different revision")
    if not re.fullmatch(r"[0-9a-f]{40}", rev):
        raise ValueError("HF metadata lacks an immutable revision")
    base = f"https://huggingface.co/{repo}/resolve/{rev}/"
    config_data = _fetch_small(base + "config.json")
    card_data = _fetch_small(base + "README.md")
    config = json.loads(config_data)
    text = config.get("text_config") or config
    files, sizes, evidence = {}, {}, {}
    for entry in metadata["siblings"]:
        path = entry["rfilename"]
        if not _safe_path(path):
            raise ValueError("unsafe path in model metadata")
        if not _artifact(path):
            continue
        lfs = entry.get("lfs") or {}
        if lfs.get("sha256"):
            digest, method = lfs["sha256"], "HF LFS SHA-256 metadata (body not downloaded)"
        else:
            if path.endswith(".safetensors"):
                raise ValueError("weight checksum absent from LFS metadata; refusing body download")
            data = config_data if path == "config.json" else _fetch_small(base + urllib.parse.quote(path))
            digest, method = hashlib.sha256(data).hexdigest(), "SHA-256 of fetched small file"
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid SHA-256 in HF metadata")
        files[path] = digest
        sizes[path] = entry.get("size", lfs.get("size"))
        evidence[path] = method
    n = int(text["num_hidden_layers"])
    width = int(text["intermediate_size"])
    shared = int(text.get("num_kv_shared_layers", 0))
    double = bool(text.get("use_double_wide_mlp", False))
    widths = [width * (2 if double and i >= n - shared else 1) for i in range(n)]
    qat = "-qat" in name
    assistant = name.endswith("-assistant")
    role = "assistant" if assistant else "target"
    card = metadata.get("cardData") or {}
    params = (metadata.get("safetensors") or {}).get("total")
    out = {
        "repo": repo, "revision": rev, "family": config["model_type"],
        "architecture": config["architectures"][0], "role": role,
        "target": name.removesuffix("-assistant") if assistant else None,
        "layers": n, "hidden": int(text["hidden_size"]), "widths": widths,
        "layer_types": text["layer_types"], "sliding_window": text.get("sliding_window"),
        "max_position_embeddings": text["max_position_embeddings"],
        "vocab_size": text["vocab_size"], "num_kv_shared_layers": shared,
        "hidden_size_per_layer_input": text.get("hidden_size_per_layer_input", 0),
        "final_logit_softcapping": text.get("final_logit_softcapping"),
        "dtype": "float16" if qat else "float32", "quantized": qat,
        "forward_only": qat, "gradient_measurements": not qat and not assistant,
        "license": card.get("license"), "gated": metadata.get("gated"),
        "license_url": card.get("license_link"),
        "parameters": params if not qat else None,
        "stored_tensor_metadata": metadata.get("safetensors"),
        "weight_bytes": sum(size for path, size in sizes.items() if path.endswith(".safetensors")),
        "sha256": files, "file_bytes": sizes, "checksum_evidence": evidence,
        "card_sha256": hashlib.sha256(card_data).hexdigest(),
        "sources": {"metadata": f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true", "config": base + "config.json",
                    "card": f"https://huggingface.co/{repo}/blob/{rev}/README.md"},
    }
    _validate_spec(name, out)
    return out


def _validate_spec(name: str, s: dict) -> None:
    if s["repo"] != MODEL_REPOS[name] or not re.fullmatch(r"[0-9a-f]{40}", s["revision"]):
        raise ValueError(f"{name}: repo/revision mismatch")
    if len(s["widths"]) != s["layers"] or len(s["layer_types"]) != s["layers"]:
        raise ValueError(f"{name}: inconsistent layer layout")
    if min(s["widths"]) < 1 or s["hidden"] < 1:
        raise ValueError(f"{name}: invalid dimensions")
    if s["family"] not in ("gemma4", "gemma4_unified", "gemma4_assistant"):
        raise ValueError(f"{name}: unreviewed architecture")
    if s["license"] != "apache-2.0":
        raise ValueError(f"{name}: licence requires a new review")
    if s["role"] == "assistant" and s["target"] not in ("gemma4-e2b", "gemma4-e4b"):
        raise ValueError(f"{name}: unreviewed assistant pairing")
    required = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"}
    if not required <= s["sha256"].keys() or not any(p.endswith(".safetensors") for p in s["sha256"]):
        raise ValueError(f"{name}: incomplete file manifest")
    for path, digest in s["sha256"].items():
        if not _safe_path(path) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{name}: unsafe file or checksum")


def check() -> dict:
    """Validate manifest structure offline; not a runtime/weight verification."""
    if set(MODEL_SPECS) != set(MODEL_REPOS):
        raise ValueError("model registry must contain all eight audited models")
    for name, s in MODEL_SPECS.items():
        _validate_spec(name, s)
    return {"ok": True, "models": len(MODEL_SPECS), "audit": "offline manifest structure",
            "weights_downloaded": False, "gpu_runtime_verified": False}


def _write_report(report: dict, output: str | None) -> None:
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(encoded, end="")
        return
    dest = Path(output).expanduser().resolve()
    repo = Path(__file__).resolve().parents[3]
    if dest == repo or repo in dest.parents:
        raise ValueError("audit reports must be written outside the repository")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: a refresh must not silently replace prior audit evidence.
    with dest.open("x", encoding="utf-8") as f:
        f.write(encoded)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("check", "refresh"))
    p.add_argument("--model", action="append", choices=tuple(MODEL_REPOS))
    p.add_argument("--pinned", action="store_true", help="audit recorded revisions instead of current main")
    p.add_argument("--output", help="new JSON file outside the repo (default: stdout)")
    a = p.parse_args(argv)
    if a.command == "check":
        report = check()
    else:
        names = a.model or list(MODEL_REPOS)
        models = {n: audit_model(n, MODEL_SPECS[n]["revision"] if a.pinned else None) for n in names}
        report = {"audited_at": dt.datetime.now(dt.timezone.utc).isoformat(), "models": models,
                  "changed": [n for n in names if models[n] != MODEL_SPECS[n]],
                  "weights_downloaded": False, "gpu_runtime_verified": False}
    _write_report(report, a.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
