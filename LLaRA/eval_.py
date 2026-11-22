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
    # "1": "runs/mid_data/results/832b7bfab9e9d3753024a9afcd8e4053",
    # "2": "runs/mid_data/results/34cfe5bd7e1ec44821487450dd5d8e35",
    "3": "runs/mid_data/results/75429e72820e7dece5cbefe2ed51b701",
    "4": "runs/mid_data/results/158a83ed452e486932dc4fefe19b2575",
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
    "1": "runs/small_data/results/b00a7aecd373aedac1314c16e65105f4",
    "2": "runs/small_data/results/c8db89f5d039e16e7fb770af5e049677",
    "3": "runs/small_data/results/9c00629b1603ad85a72e17125065c236",
    "4": "runs/small_data/results/020f23daf457d6f3045d36c6389a126b",
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
for i in range(2):
    run_dir = exec_config[str(i + 3)]
    if not run_dir:  
        continue

    code[20] = f"test_negative = pkl.load(open('{exec_config['test_negative']}', 'rb'))  #\n"
    code[32] = f"    lora_weights_path: str = '{run_dir}'   \n"
    # code[47] = f"    seed: int = {exec_config[f"seed_{i + 1}"]}\n"
    code[132] = f"    id2title_path = '{exec_config['id2title_path']}'\n"

    # Emb_v / Emb_k / user_features 必须用该 seed 自己那套
    prepare_dir = exec_config[f"dir_{i + 1}"]
    code[141] = f"    k_embs = np.load('{prepare_dir}/{exec_config['k_embs']}')   #\n"

    code[142] = f"    vk_dict = pkl.load(open('{exec_config['vk_dict']}', 'rb'))\n"
    code[145] = f"        data_path='{exec_config['train_data_path']}',\n"

    exec("".join(code))
    # print("".join(code))
    # break
    print(f"{i + 1}: done")