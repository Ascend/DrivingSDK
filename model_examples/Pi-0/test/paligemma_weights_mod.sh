paligemma_weights=$1
sed -i "251s|self.language_tokenizer = AutoTokenizer.from_pretrained(\"google/paligemma-3b-pt-224\")|self.language_tokenizer = AutoTokenizer.from_pretrained(\"${paligemma_weights}\")|" lerobot/common/policies/pi0/modeling_pi0.py
