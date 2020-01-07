from components.Controller import Controller
import numpy as np

def test_all(model):
    controller = Controller()
    controller.set_model(model)
    controller.train()
    controller.test_all()
    print(controller.get_q_table())


def apply_person(model, p_id):
    controller = Controller()
    controller.set_model(model)
    controller.train()
    controller.test_person(p_id)
    print(controller.get_q_table())

    controller.set_user(p_id)
    controller.apply(p_id)
    controller.test_person(p_id)
    print(controller.get_q_table())

def apply_all(model):
    controller = Controller()
    controller.set_model(model)

    controller.train()
    q_table = np.copy(controller.get_q_table())

    total_accuracy = 0
    for p_id in range(1, 31):
        controller.reload_default_policy()
        controller.set_user(p_id)
        controller.apply(p_id)
        results = controller.test_person(p_id)
        total_accuracy += (sum(results.values()) / len(results)) / 30
    print(total_accuracy)

def main():
    p_id = 3
    model = 'QLearning'

    # apply_person(model, p_id)
    apply_all(model)

if '__main__' == __name__:
    main()
