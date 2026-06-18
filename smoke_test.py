import math
import random

# --- RC-GRPO inject_diverse_reward_tokens (local copy) ---
def inject_diverse_reward_tokens_local(dataset_items, num_generations: int = 8, high_token: str = "<|high_reward|>", low_token: str = "<|low_reward|>", high_fraction: float = 0.5):
    n_high = max(1, round(num_generations * high_fraction))
    out = []
    for idx, example in enumerate(dataset_items):
        group = []
        for g in range(num_generations):
            ex = dict(example)
            token = high_token if g < n_high else low_token
            if not ex['prompt'].startswith(token):
                ex['prompt'] = token + "\n" + ex['prompt']
            ex['reward_token'] = token
            group.append(ex)
        rnd = random.Random(int(idx))
        rnd.shuffle(group)
        out.extend(group)
    return out

# --- AWPO compute_advantages (local copy) ---
def compute_awpo_advantages_local(rewards_list, group_indices_list, completions=None, reasoning_cache=None, clip_shrink_coeff=0.2):
    eps = 1e-8
    B = len(rewards_list)
    advantages = [0.0] * B
    mixing_weights = []
    # build groups
    groups = {}
    for i, gid in enumerate(group_indices_list):
        groups.setdefault(gid, []).append(i)
    for gid, g_idx in groups.items():
        out_r = [float(rewards_list[i]) for i in g_idx]
        if completions is not None and reasoning_cache is not None:
            reason_r = [reasoning_cache.get(completions[i], 0.0) for i in g_idx]
        else:
            reason_r = [0.0] * len(g_idx)
        n = len(out_r)
        mu_out = sum(out_r) / n
        sigma2_out = sum((r - mu_out) ** 2 for r in out_r) / n
        sigma_out = math.sqrt(sigma2_out + eps)
        rho = max(0.0, min(1.0, 0.5 * (1.0 - sigma_out)))
        for _ in range(10):
            mixed = [(1 - rho) * r_out + rho * r_rea for r_out, r_rea in zip(out_r, reason_r)]
            mu_mix = sum(mixed) / n
            sigma_mix = math.sqrt(sum((r - mu_mix) ** 2 for r in mixed) / n + eps)
            new_rho = sigma_mix / (sigma_out + sigma_mix + eps)
            if abs(new_rho - rho) < 1e-4:
                rho = new_rho
                break
            rho = new_rho
        w = 4.0 * mu_out * (1.0 - mu_out)
        mixed = [(1 - rho) * r_out + rho * r_rea for r_out, r_rea in zip(out_r, reason_r)]
        mu_mix = sum(mixed) / n
        sigma_mix = math.sqrt(sum((r - mu_mix) ** 2 for r in mixed) / n + eps)
        sigma_mix = max(sigma_mix, eps)
        group_advs = [w * (r - mu_mix) / sigma_mix for r in mixed]
        for local_idx, global_idx in enumerate(g_idx):
            advantages[global_idx] = group_advs[local_idx]
        mixing_weights.append(rho)
    avg_rho = sum(mixing_weights) / max(len(mixing_weights), 1)
    return advantages, avg_rho

# --- AutoTool token registration simulation ---
def autotool_register_simulation(existing_tokens=None):
    special = ["<reasoning>", "</reasoning>", "<tool_call>", "</tool_call>"]
    existing = set(existing_tokens or [])
    to_add = [t for t in special if t not in existing]
    return to_add

# --- Run tests ---
print('Running smoke tests (local implementations)')
# RC-GRPO test
items = [{'prompt':'original prompt','ground_truth':{}}]
expanded = inject_diverse_reward_tokens_local(items, num_generations=4, high_fraction=0.5)
print('RC expanded count:', len(expanded))
print('RC tokens:', [e['reward_token'] for e in expanded])
# AWPO test
rewards = [1.0, 0.0, 1.0, 0.0]
group_indices = [0,0,1,1]
completions = ['c1','c2','c3','c4']
reasoning_cache = {'c1':0.2,'c2':0.1,'c3':0.3,'c4':0.0}
advs, rho = compute_awpo_advantages_local(rewards, group_indices, completions, reasoning_cache)
print('AWPO advantages:', advs)
print('AWPO avg rho:', rho)
# AutoTool test
added = autotool_register_simulation(existing_tokens=['<s>'])
print('AutoTool tokens to add:', added)
print('Smoke tests completed successfully')
