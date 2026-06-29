results = """student_Original_nemotron_stem_llama8bins_opus_Arabic	embed_sim=39.6000	rouge_l=20.6000	judge=42.8000
student_Original_nemotron_stem_llama8bins_opus_French	embed_sim=50.0000	rouge_l=32.8000	judge=69.6000
student_Original_nemotron_stem_llama8bins_opus_German	embed_sim=47.0000	rouge_l=27.6000	judge=54.0000
student_Original_nemotron_stem_llama8bins_opus_Italian	embed_sim=43.6000	rouge_l=26.0000	judge=61.6000
student_Original_nemotron_stem_llama8bins_opus_Japanese	embed_sim=34.4000	rouge_l=11.4000	judge=34.8000
student_Original_nemotron_stem_llama8bins_opus_Portuguese	embed_sim=51.0000	rouge_l=32.8000	judge=66.8000
student_Original_nemotron_stem_llama8bins_opus_Spanish	embed_sim=53.4000	rouge_l=36.4000	judge=68.2000

student_Original_nemotron_stem_llama8bins_mmmlu_AR_XY	embed_sim=41.8000	rouge_l=41.8000	judge=42.6000
student_Original_nemotron_stem_llama8bins_mmmlu_DE_DE	embed_sim=52.0000	rouge_l=52.0000	judge=54.0000
student_Original_nemotron_stem_llama8bins_mmmlu_ES_LA	embed_sim=51.8000	rouge_l=51.8000	judge=55.2000
student_Original_nemotron_stem_llama8bins_mmmlu_FR_FR	embed_sim=53.2000	rouge_l=53.2000	judge=57.8000
student_Original_nemotron_stem_llama8bins_mmmlu_IT_IT	embed_sim=50.6000	rouge_l=50.6000	judge=55.0000
student_Original_nemotron_stem_llama8bins_mmmlu_JA_JP	embed_sim=41.4000	rouge_l=41.4000	judge=44.4000
student_Original_nemotron_stem_llama8bins_mmmlu_PT_BR	embed_sim=54.0000	rouge_l=54.0000	judge=58.8000

student_Original_nemotron_stem_llama8bins_mhotpot_ar	embed_sim=61.4000	rouge_l=3.2000	judge=22.0000
student_Original_nemotron_stem_llama8bins_mhotpot_en	embed_sim=66.2000	rouge_l=52.2000	judge=54.2000
student_Original_nemotron_stem_llama8bins_mhotpot_ru	embed_sim=61.4000	rouge_l=10.4000	judge=39.8000
student_Original_nemotron_stem_llama8bins_mhotpot_zh	embed_sim=59.2000	rouge_l=8.4000	judge=28.0000

student_Original_nemotron_stem_llama8bins_nemotron_stem	embed_sim=55.8000	rouge_l=56.0000	judge=59.6000
student_Original_nemotron_stem_llama8bins_nemotron_math	embed_sim=26.8000	rouge_l=7.4000	judge=3.8000
student_Original_nemotron_stem_llama8bins_nemotron_chat	embed_sim=18.0000	rouge_l=0.6000	judge=87.8000"""
import time
lines = results.split("\n")
print(3)
time.sleep(1)
print(2)
time.sleep(1)
print(1)
time.sleep(1)
for line in lines:
    if len(line) < 5:
        print("\n\n------- new -------\n\n")
        time.sleep(7)
        print(3)
        time.sleep(1)
        print(2)
        time.sleep(1)
        print(1)
        time.sleep(1)

    else:
        data = line.split("\t")
        print(data[0])
        for d in data[1:]:
            print(d)
            time.sleep(3)
