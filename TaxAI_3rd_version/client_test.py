from client import *
if __name__ == "__main__":
    # 示例用法
    initial_communicate_with_server(USER_ID)
    code_path = "TaxAI_3rd_version/agents/model_self"
    # fetch_random_top_k_model(user_id=USER_ID, dest_dir= code_path)
    push_folder("/home/mhm/workspace/Competition_TaxingAI/TaxAI_3rd_version/agents/model_self", user_id=USER_ID, model_id="test_model4", algo_name="ppo", epoch=0)

    # fetch_random_models(user_id=USER_ID, dest_dir= "TaxAI_3rd_version/agents/model_self")
#     fetch_random_top_k_model(user_id=USER_ID, dest_dir="TaxAI_3rd_version/agents/model_self")