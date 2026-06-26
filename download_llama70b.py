#!/usr/bin/env python3
# ==============================================================================
# Script: download_llama33_70b.py
# Descrição: Baixa os pesos oficiais do Llama-3.3-70B-Instruct.
# ==============================================================================

import os
from huggingface_hub import snapshot_download

# Atualizado cirurgicamente para a versão 3.3 escolhida por você
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
LOCAL_DIR = os.path.expanduser("~/qcom_model_workspace/models/Llama-3.3-70B-Instruct")

print(f"=== Iniciando Esteira de Download: {MODEL_ID} ===")
print(f"Destino local: {LOCAL_DIR}")

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("\n[ERRO] Variável de ambiente HF_TOKEN não encontrada!")
    print("Por favor, exporte seu token antes de rodar o script:")
    print("export HF_TOKEN='seu_token_aqui'\n")
    exit(1)

try:
    print("\nConectando ao Hugging Face Hub (Llama 3.3) e calculando blocos...")
    # Faz o download otimizado ignorando arquivos de terceiros (como GGUF)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        token=hf_token,
        ignore_patterns=["*.gguf", "*.onnx", "*.pki"],
        max_workers=4  # Divide o download em 4 threads paralelas
    )
    print(f"\n[SUCESSO] Todos os arquivos do Llama-3.3-70B foram salvos em: {LOCAL_DIR}")

except Exception as e:
    print(f"\n[ERRO] Falha ao realizar o download do modelo: {e}")
    exit(1)
