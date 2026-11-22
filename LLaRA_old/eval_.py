mid_data = {
    "seed_1": 12,
    "seed_2": 22,
    "seed_3": 32,
    "seed_4": 42,
    "seed_5": 52,
    "dir_path": "LLaRA_old/runs/mid_data/results",

    # "dir_1": "f04e2f186abc3d328cb6a868fb82461d",
    "dir_2": "2ebfa4ffa3d9e404bb9bbd9dcf4388fc",
    # "dir_3": "c6bd1f7335bae938b1d4d717fdb68215",
    # "dir_4": "0ace0b5d6e7d9b296238e33d1bc68547",
    "dir_5": "315110338dd9d2f3888d346056fa8b90",
    
    "sasrec_path": "sasrec_old/runs/mid_data",
    "1": "1b179df098b8ad3801d17f65cc105291",
    "2": "742474b1b5911a90cf0b31d303c78db6",
    "3": "05b5d096e123b7b89cf4ccfaed200fad",
    "4": "116719ce0d1bed7972d4defb6130686c",
    "5": "d807deb4019e7acafd77f5c197cc2d2f",
    "concept_csv": "data/MOOCCube/mid_data/need_data/entities/concepts.csv",

    "concept_data": "prepare_data/mooc_sasrec_augllm2/test_concept_data.p",

    "cand_source": "align_masked",

}
small_data = {
    "seed_1": 12,
    "seed_2": 22,
    "seed_3": 32,
    "seed_4": 42,
    "seed_5": 52,
    "dir_path": "LLaRA_old/runs/small_data/results",

    "dir_1": "a68598a6474c37a6dd531b52e5612c09",
    "dir_2": "65b404a33c5678d07df5f938a500ab3f",
    "dir_3": "20db4433c10e8640947f32e66312c2cd",
    "dir_4": "db791e0f0c2cf7c779883f298dc65bdd",
    "dir_5": "798cd7d0344e00ddfc8634dce2638db7",
    
    "sasrec_path": "sasrec_old/runs/small_data",
    "1": "bf5ca240921cb8205da27221ce68950b",
    "2": "09a6e42b38b5b57ca8ed691e4d969b6a",
    "3": "4610653b6c0367ff2c795f2e91cee4b9",
    "4": "247d557ba6e95614a4b1f1b47e238f18",
    "5": "ed6b9c77f758808ff10844b06e6424ff",
    "concept_csv": "data/MOOCCube/small_data/need_data/entities/concepts.csv",

    "concept_data": "prepare_data/mooc_sasrec_augllm/test_concept_data.p",

    "cand_source": "align_masked",

}

with open("infer_mooc_llara_old.py", "r", encoding="utf-8-sig") as f:  
    code = f.readlines()

exec_config = mid_data
for i in range(5):
    # 如果不是0和3，就跳过
    if i != 1 and i != 4:
        continue
    run_dir = exec_config[str(i + 1)]
    if not run_dir:  
        continue

    code[46] = f"LORA_DIR = os.path.join(ROOT, '{exec_config['dir_path']}/{exec_config[f'dir_{i + 1}']}')\n"
    code[48] = f"CONCEPT_CSV = os.path.join(ROOT, '{exec_config['concept_csv']}')\n"
    code[50] = f"K_EMB_PATH = os.path.join(ROOT, '{exec_config['sasrec_path']}/prepare_data/{exec_config[f'{i + 1}']}/Emb_k.npy')\n"
    code[51] = f"TEST_DATA = os.path.join(ROOT, '{exec_config['concept_data']}')\n"
    code[54] = f"CAND_SOURCE = '{exec_config['cand_source']}'\n"
    code[55] = f"CAND_PKL = os.path.join(ROOT, '{exec_config['sasrec_path']}/results/{exec_config[f'{i + 1}']}/top20_best_sasrec_old_test_masked.pkl')\n"

    code[74] = f"    seed: int = {exec_config[f'seed_{i + 1}']} \n"

    exec("".join(code))
    print(f"{i + 1}: done")