# skill-tree v2

單一資料來源：`tree_data.py`（META + 節點）＋ `quiz_a*.py`（題庫，經 `quiz_bank.py` 彙總）。

```bash
python3 build.py   # 驗證 + 產生 ros2-platform.html（單檔、可直接發布成 Artifact）
```

驗證不過會 exit 1（重複 id、依賴循環、dod 不可判定、題庫長度洩漏、答案分佈失衡…），不要繞過去手改產物。
進度存 localStorage（key: ros2-platform-skilltree-v2），用頁面的「進度」對話框備份/還原。
