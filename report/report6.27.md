## 6.27工作报告

### 工作内容
1. 重新配置工作环境
2. 重写了服务器与客户端对于log中path的记录维护，使其与之前版本一致（即path值为形如`test_model4,Server/policy_pools/test2_test_model4/run/house_net.pt`的字符串）
3. 继续调试训练时引入其他模型的过程