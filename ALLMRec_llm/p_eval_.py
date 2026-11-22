mid_data = {
    "seed_1": 12,
    "seed_2": 22,
    "seed_3": 32,
    "seed_4": 42,
    "seed_5": 52,
    "dir_path": "ALLMRec_llm/runs/mid_data/results",

    "dir_1": "20be55c4f4db2d1e33933a7b71763eb2",
    "dir_2": "144e8e0264dc4796257feca90c4485cd",
    "dir_3": "1d431bbc072969f77225b4bd72714ce8",
    # "dir_4": "0ace0b5d6e7d9b296238e33d1bc68547",
    # "dir_5": "315110338dd9d2f3888d346056fa8b90",
    
    "sasrec_path": "ALLMRec_old/runs/mid_data",
    "1": "fe6f92622c461b2216f8bf1a215da13f",
    "2": "6fa7f48b48ac46366199e773be47fca7",
    "3": "e55a6c142a6a3366f7a169d0bdacaf7c",
    "4": "877d74f25f2e7384ab16648ec9b3c997",
    "5": "a2d4f1011d91615d7ddc8f7478551f53",
    "concept_csv": "data/MOOCCube/mid_data/need_data/entities/concepts.csv",

    "concept_data": "prepare_data/mooc_sasrec_augllm2/test_concept_data.p",

    "cand_source": "sasrec_global",

}
small_data = {
    "seed_1": 12,
    "seed_2": 22,
    "seed_3": 32,
    "seed_4": 42,
    "seed_5": 52,
    "dir_path": "ALLMRec_llm/runs/small_data/results",

    "dir_1": "48f1213455123e73f246c39af55511f1",
    "dir_2": "1cb9b9d39e420be7f3f0e069095f911b",
    # "dir_3": "20db4433c10e8640947f32e66312c2cd",
    "dir_4": "f869b292db1d3073d7ff2c1c92ac24fe",
    # "dir_5": "798cd7d0344e00ddfc8634dce2638db7",
    
    "sasrec_path": "ALLMRec_old/runs/small_data",
    "1": "14fa85141c1899b3888451faf6dc146d",
    "2": "9a4320be14bd686015d193ca4fe14b69",
    "3": "5bcdcced83d6d067651637c29056f320",
    "4": "3cd4865a871d47fa9f4c521b1d4f9443",
    "5": "08dae6f88e5328559dcf58046a4bf768",
    "concept_csv": "data/MOOCCube/small_data/need_data/entities/concepts.csv",

    "concept_data": "prepare_data/mooc_sasrec_augllm/test_concept_data.p",

    "cand_source": "sasrec_global",

}

with open("infer_mooc_llm.py", "r", encoding="utf-8-sig") as f:  
    code = f.readlines()

exec_config = mid_data
for i in range(5):
    # 如果不是0和3，就跳过
    if i == 0 or i == 3:
        continue
    run_dir = exec_config[str(i + 1)]
    if not run_dir:  
        continue

    code[48] = f"ADAPTOR_DIR = os.path.join(ROOT, '{exec_config['dir_path']}/{exec_config[f'dir_{i + 1}']}')\n"
    code[50] = f"CONCEPT_CSV = os.path.join(ROOT, '{exec_config['concept_csv']}')\n"
    code[52] = f"STAGE1_DIR = os.path.join(ROOT, '{exec_config['sasrec_path']}/prepare_data/{exec_config[f'{i + 1}']}')\n"
    code[55] = f"TEST_DATA = os.path.join(ROOT, '{exec_config['concept_data']}')\n"
    code[62] = f"CAND_SOURCE = '{exec_config['cand_source']}'\n"
    code[63] = f"CAND_PKL = os.path.join(ROOT, '{exec_config['sasrec_path']}/results/{exec_config[f'{i + 1}']}/top20_best_sasrec_old_test_global.pkl')\n"

    code[94] = f"    seed: int = {exec_config[f'seed_{i + 1}']} \n"

    exec("".join(code))
    print(f"{i + 1}: done")