class Event:
    def __init__(self, event_id: str, options: list, body_text: str = ""):
        self.event_id = event_id
        self.options = options
        self.chosen_option = -1  # 未選択時は-1
        self.body_text = body_text

    @classmethod
    def from_json(cls, json_object):
        return cls(
            event_id=json_object["event_id"],
            options=json_object["options"],
            body_text=json_object.get("body_text", "")
        )