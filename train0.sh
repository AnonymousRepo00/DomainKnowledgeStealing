python trainStudent.py \
  --STUDENT_BASE="meta-llama/Llama-3.1-8B-Instruct" \
  --STUDENT_DEVICE="cuda:0" \
  --domain="medical" \
  --LORA_BATCH_SIZE=16 \
  --LORA_EPOCHS=4 \
  --LORA_LR=2e-4 \
  --CLUSTER_FREQ=10 \
  --MAX_QUERIES=500 \
  --sim_threshold=0.6 \
  --Teacher_model="gpt-4.1" \
  --broad_answer_thresh=0.5 \
  --CKPT_FREQ=1 \
#   --RESUME_CKPT latest \
