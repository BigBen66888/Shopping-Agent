from eval_v3.context_end_to_end import condition


def test_color_variant_in_description_counts_only_for_relevant_product():
    color = {"name": "白色", "fields": ["title", "description"], "any_terms": ["white"]}
    category = {"name": "背包", "any_terms": ["backpack"]}
    backpack = {"title": "Travel Backpack", "description": "Available in white or black"}
    shirt = {"title": "Cotton Shirt", "description": "Styled with a white backpack"}
    assert condition(backpack, color) and condition(backpack, category)
    assert condition(shirt, color) and not condition(shirt, category)


def test_membrane_keyboard_does_not_pass_mechanical_check():
    check = {"name": "机械", "any_terms": ["mechanical"],
             "reject_title_phrases": ["mechanical feel", "mechanical touch"],
             "reject_attributes": {"keyboard_switch_type": ["membrane"]}}
    membrane = {"title": "Mechanical Touch Keyboard", "attributes": {"keyboard_switch_type": ["membrane"]}}
    switches = {"title": "Mechanical Keyboard", "attributes": {"keyboard_switch_type": ["red switch"]}}
    assert not condition(membrane, check)
    assert condition(switches, check)
