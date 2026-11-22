mid_data = {
    "test_negative": "../prepare_data/mooc_sasrec_augllm2/test_new_negative.p",
    "seed_1": 12,
    "seed_2": 22,
    "seed_3": 32,
    "seed_4": 52,
    "dir_1": "../sasrec_new/runs/mid_data/prepare_data/ad7ec1abda0fd5b2a0d6e51aacf6b2b3",
    "dir_2": "../sasrec_new/runs/mid_data/prepare_data/eb4baec60d7c628ca45ff910b1279f28",
    "dir_3": "../sasrec_new/runs/mid_data/prepare_data/5b1cac37bbb0d5f86f3771a9f25338df",
    "dir_4": "../sasrec_new/runs/mid_data/prepare_data/327b92b578defd2f295a6714f83b0c60",
    "1": "runs/mid_data/results/0e4c91776329648661b6b8895e9bdec7",
    "2": "runs/mid_data/results/60d2beff0cd1ab5458c2e3de4b5801b3",
    "3": "runs/mid_data/results/092faa739b8944644a2fce2dd5aa68d2",
    "4": "runs/mid_data/results/d6578dc645ea77319be626444d8d828a",
    "id2title_path": "../data/MOOCCube/mid_data/need_data/entities/concepts.csv",
    "v_embs": "Emb_v.npy",
    "k_embs": "Emb_k.npy",
    "u_embs": "eval_user_features.npy",
    "vk_dict": "../data/MOOCCube/mid_data/need_data/relations/vk.p",
    "train_data_path": "../prepare_data/mooc_sasrec_augllm2/eval_data_new.p",

}
small_data = {
    "test_negative": "../prepare_data/mooc_sasrec_augllm/test_new_negative.p",
    "seed_1": 12,
    "seed_2": 22,
    "seed_3": 32,
    "seed_4": 52,
    "dir_1": "../sasrec_new/runs/small_data/prepare_data/39ca25ef14821014de76d4dbb3ead1b8",
    "dir_2": "../sasrec_new/runs/small_data/prepare_data/5b57dda4ac547bc823e7201d1ba5f184",
    "dir_3": "../sasrec_new/runs/small_data/prepare_data/a8d002ef9b217cff7e95b19f6564713b",
    "dir_4": "../sasrec_new/runs/small_data/prepare_data/9c6490ad686d811a0ca357becd22ab48",
    "1": "runs/small_data/results/798c547935111f783b17bbea7586b5b9",
    "2": "runs/small_data/results/dfc3ff17bc9cadd52e8dee834509b327",
    "3": "runs/small_data/results/4bfcdbf579621cd02c292884fb6ca41a",
    "4": "runs/small_data/results/ea0c7b81e573bea78105f1ef71a98e7c",
    "id2title_path": "../data/MOOCCube/small_data/need_data/entities/concepts.csv",
    "v_embs": "Emb_v.npy",
    "k_embs": "Emb_k.npy",
    "u_embs": "eval_user_features.npy",
    "vk_dict": "../data/MOOCCube/small_data/need_data/relations/vk.p",
    "train_data_path": "../prepare_data/eval_data_new.p",
}

with open("eval_mooc_base.py", "r", encoding="utf-8-sig") as f:  
    code = f.readlines()

META_KEYS = ("test_negative", "id2title_path", "v_embs", "k_embs", "u_embs",
             "vk_dict", "train_data_path")

exec_config = mid_data
for i in range(4):
    run_dir = exec_config[str(i + 1)]
    if not run_dir:  
        continue

    code[23] = f"test_negative = pkl.load(open('{exec_config['test_negative']}', 'rb'))  #\n"
    code[35] = f"    lora_weights_path: str = '{run_dir}'   \n"
    # code[52] = f"    seed: int = {exec_config[f"seed_{i + 1}"]}\n"
    code[153] = f"    id2title_path = '{exec_config['id2title_path']}'\n"

    # Emb_v / Emb_k / user_features 必须用该 seed 自己那套
    prepare_dir = exec_config[f"dir_{i + 1}"]
    code[164] = f"    v_embs = np.load('{prepare_dir}/{exec_config['v_embs']}')   #\n"
    code[165] = f"    k_embs = np.load('{prepare_dir}/{exec_config['k_embs']}')   #\n"
    code[166] = f"    u_embs = np.load('{prepare_dir}/{exec_config['u_embs']}')  #\n"

    code[162] = f"    vk_dict = pkl.load(open('{exec_config['vk_dict']}', 'rb'))\n"
    code[169] = f"        train_data_path='{exec_config['train_data_path']}',\n"

    exec("".join(code))
    # print("".join(code))
    # break
    print(f"{i + 1}: done")