#!/usr/bin/env python
"""
post_comments.py — Posts AI review comments to a GitHub Pull Request

Usage:
  python post_comments.py review.json

Environment:
  GITHUB_TOKEN: GitHub token with repo access
  GITHUB_REPOSITORY: owner/repo (optional, will try to parse from review.json if not set)
  GITHUB_PR_NUMBER: PR number (optional, will try to parse from review.json if not set)
"""
import json
import os
import sys
import requests

GITHUB_API = "https://api.github.com"

def main():
    if len(sys.argv) < 2:
        print("Usage: python post_comments.py review.json", file=sys.stderr)
        sys.exit(1)
    review_path = sys.argv[1]
    with open(review_path, encoding="utf-8") as f:
        reviews = json.load(f)

    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    pr_number = os.environ.get("GITHUB_PR_NUMBER")

    # Try to get repo/pr from GitHub Actions env if not set
    if not repo:
        repo = os.environ.get("GITHUB_REPOSITORY")
    if not pr_number:
        pr_number = os.environ.get("GITHUB_PR_NUMBER")
    # Fallback: try to get from workflow env
    if not repo:
        repo = os.environ.get("GITHUB_REPOSITORY")
    if not pr_number:
        pr_number = os.environ.get("GITHUB_REF", "").split("/")[-1]

    if not token or not repo or not pr_number:
        print("Missing GITHUB_TOKEN, GITHUB_REPOSITORY, or GITHUB_PR_NUMBER", file=sys.stderr)
        sys.exit(1)

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }

    # Post each review comment
    for item in reviews:
        body = f"**{item['severity']}**: {item['explanation']}\n\n**Suggestion:** {item['suggestion']}"
        data = {
            "body": body,
            "commit_id": None,  # Let GitHub use the latest commit
            "path": item["file"],
            "line": item["line"],
            "side": "RIGHT"
        }
        # Get latest commit SHA for the PR
        pr_url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}"
        pr_resp = requests.get(pr_url, headers=headers)
        pr_resp.raise_for_status()
        pr_data = pr_resp.json()
        commit_id = pr_data["head"]["sha"]
        data["commit_id"] = commit_id
        # Post the comment
        url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/comments"
        resp = requests.post(url, headers=headers, json=data)
        if resp.status_code >= 300:
            print(f"Failed to post comment: {resp.text}", file=sys.stderr)
        else:
            print(f"Posted comment on {item['file']}:{item['line']}")

if __name__ == "__main__":
    main()
