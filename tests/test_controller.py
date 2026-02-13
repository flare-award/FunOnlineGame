from autopilot_bot.controller import InputController


class FakeKeyboard:
    def __init__(self):
        self.pressed = []
        self.released = []

    def press(self, key):
        self.pressed.append(key)

    def release(self, key):
        self.released.append(key)


def test_send_action(monkeypatch):
    fake = FakeKeyboard()
    monkeypatch.setattr("autopilot_bot.controller.keyboard", fake)

    ctrl = InputController({"forward": "w"}, anti_spam_ms=0)
    assert ctrl.send_action("forward") is True
    assert fake.pressed == ["w"]
    assert fake.released == ["w"]


def test_send_action_unknown(monkeypatch):
    fake = FakeKeyboard()
    monkeypatch.setattr("autopilot_bot.controller.keyboard", fake)

    ctrl = InputController({"forward": "w"}, anti_spam_ms=0)
    assert ctrl.send_action("jump") is False
