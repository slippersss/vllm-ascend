import requests

from prometheus_client.parser import text_string_to_metric_families

HOST_IP = "xxx"
NUM_SPEC = 3


def analysis_metrics(metrics_text: str, num_speculative_tokens: int) -> tuple[int, list[int]]:
    num_drafts = 0
    num_accepted_tokens_per_pos = [0] * num_speculative_tokens
    for family in text_string_to_metric_families(metrics_text):
        if family.name == "vllm:spec_decode_num_drafts":
            assert family.type == "counter"
            for sample in family.samples:
                num_drafts += sample.value
        elif family.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert family.type == "counter"
            for sample in family.samples:
                pos = int(sample.labels["position"])
                num_accepted_tokens_per_pos[pos] += sample.value
    return num_drafts, num_accepted_tokens_per_pos


if __name__ == "__main__":
    try:
        req = requests.get(f"http://{HOST_IP}:60006/metrics")
        if req.status_code == 200:
            num_drafts, num_accepted_tokens_per_pos = analysis_metrics(req.text, NUM_SPEC)
            if num_drafts > 0:
                acceptance_per_pos = [num_accepted_tokens / num_drafts for num_accepted_tokens in num_accepted_tokens_per_pos]
                acceptance_len = 1 + sum(acceptance_per_pos)
                print(f"{num_drafts=}")
                print(f"{num_accepted_tokens_per_pos=}")
                print(f"{acceptance_per_pos=}")
                print(f"{acceptance_len=}")
            else:
                print("No draft")
        else:
            print("No response")
    except:
        print("No link")
