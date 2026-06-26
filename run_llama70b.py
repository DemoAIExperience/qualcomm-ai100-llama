from QEfficient import QEFFAutoModelForCausalLM
from transformers import AutoTokenizer

model_name = "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4"

print("Carregando e otimizando o modelo (vai baixar ~40GB na primeira vez)...")
model = QEFFAutoModelForCausalLM.from_pretrained(model_name)

print("Compilando para a AI100 (pode levar bastante tempo num 70B)...")
qpc_path = model.compile(
    num_devices=2,        # seus 2 QIDs
    num_cores=16,          # 16 NSPs por QID, confirmado no qaic-util
    prefill_seq_len=128,
    ctx_len=4096,
    batch_size=1,
    mxint8_kv_cache=True,  # compressão da KV cache, margem de memória extra
)
print(f"QPC gerado em: {qpc_path}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model.generate(
    tokenizer=tokenizer,
    prompts=["Explique em uma frase o que é a Qualcomm Cloud AI 100."],
    device_id=[0, 1],
)
