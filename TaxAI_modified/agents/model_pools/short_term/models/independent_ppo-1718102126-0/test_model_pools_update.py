
from model_pools_update import *
import random

epoch = 50

# 生成一组随机的government_score和household_score
gov_scores = [random.randint(0, 100) for i in range(epoch)]
household_scores = [random.randint(0, 100) for i in range(epoch)]

short_pool_size = 5
long_pool_size = 30

top_k_pool_size = 10



# 找到这些gov_scores、household_scores中的top_k项，并标记出哪一轮出现

def find_top_k(scores, k):
    top_k_scores = sorted(scores, reverse=True)[:k]
    top_k_index = [scores.index(i) for i in top_k_scores]
    return top_k_scores, top_k_index
top_k_gov_score, top_k_gov_index = find_top_k(gov_scores, top_k_pool_size)
top_k_household_score, top_k_household_index = find_top_k(household_scores, top_k_pool_size)

print("Top k government scores and indexes: ", top_k_gov_score, top_k_gov_index)
print("Top k household scores and indexes: ", top_k_household_score, top_k_household_index)

for i in range(epoch):
    update_short_term_policy_pool(short_pool_size=short_pool_size, epoch=i, government_score=gov_scores[i], household_score=household_scores[i])
    update_long_term_policy_pool(long_pool_size=long_pool_size, epoch=i, government_score=gov_scores[i], household_score=household_scores[i])
    update_top_k_policy_pool(pool_size=top_k_pool_size, epoch=i, government_score=gov_scores[i], household_score=household_scores[i])

