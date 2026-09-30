from shopping_agent.database import Database
from shopping_agent.service import ShoppingService


def test_new_memory_supersedes_same_subject_slot_and_keeps_history(tmp_path):
    service = ShoppingService()
    service.db = Database(tmp_path / "memory.sqlite3")
    sid = service.create_session(user_id="conflict_test")["id"]
    old = service.remember(sid, "attribute", "用户只买黑色背包")
    other = service.remember(sid, "attribute", "用户只买黑色运动鞋")
    new = service.remember(sid, "attribute", "用户改为只买白色背包")
    active = service.long_term_memory(sid)
    assert [item["id"] for item in active] == [new["id"], other["id"]]
    with service.db.connect() as db:
        assert db.execute("SELECT status FROM memory WHERE id=?", (old["id"],)).fetchone()[0] == "SUPERSEDED"
    restored = service.remember(sid, "attribute", "用户只买黑色背包")
    assert restored["id"] == old["id"]
    assert [item["id"] for item in service.long_term_memory(sid)] == [old["id"], other["id"]]
    assert new["id"] not in {item["id"] for item in service.long_term_memory(sid)}


def test_recent_brand_and_budget_replace_only_matching_subject(tmp_path):
    service = ShoppingService()
    service.db = Database(tmp_path / "memory.sqlite3")
    sid = service.create_session(user_id="conflict_test")["id"]
    service.remember(sid, "brand", "用户长期偏好罗技鼠标")
    service.remember(sid, "brand", "用户长期偏好索尼耳机")
    service.remember(sid, "brand", "用户改为偏好雷蛇鼠标")
    service.remember(sid, "budget", "用户通常购买 300 元以下的耳机")
    service.remember(sid, "budget", "用户以后耳机预算为 500 元")
    contents = [item["content"] for item in service.long_term_memory(sid)]
    assert "用户长期偏好罗技鼠标" not in contents
    assert "用户长期偏好索尼耳机" in contents
    assert "用户改为偏好雷蛇鼠标" in contents
    assert "用户通常购买 300 元以下的耳机" not in contents
    assert "用户以后耳机预算为 500 元" in contents


def test_conflict_uses_new_value_when_old_value_is_mentioned_in_correction(tmp_path):
    service = ShoppingService()
    service.db = Database(tmp_path / "memory.sqlite3")
    sid = service.create_session(user_id="correction_test")["id"]
    service.remember(sid, "attribute", "用户只买黑色背包")
    service.remember(sid, "attribute", "用户背包改成白色，不再买黑色")
    service.remember(sid, "brand", "用户偏好罗技鼠标")
    service.remember(sid, "brand", "用户鼠标从罗技改成雷蛇")
    contents = [item["content"] for item in service.long_term_memory(sid)]
    assert contents == ["用户鼠标从罗技改成雷蛇", "用户背包改成白色，不再买黑色"]
