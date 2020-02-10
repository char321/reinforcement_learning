from components.Controller import Controller
import numpy as np
import pprint

def test_all(model):
    controller = Controller()
    controller.set_model(model)
    controller.train()
    controller.test_all()
    print(controller.get_q_table())


def apply_person(model, p_id, noi, nop, reward_scale):
    controller = Controller()
    controller.set_model(model)
    controller.train()
    controller.test_person(p_id)
    # print(controller.get_q_table())
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(controller.data[p_id])

    controller.set_user(p_id)
    controller.apply(p_id, 50, 1)
    controller.test_person(p_id)
    # print(controller.get_q_table())

def apply_all(model, noi, nop, reward_scale):
    controller = Controller()
    controller.set_model(model)

    controller.train(noi)
    q_table = np.copy(controller.get_q_table())

    total_accuracy = 0
    for p_id in range(1, 31):
        controller.reload_default_policy()
        controller.set_user(p_id)
        controller.apply(p_id, nop, reward_scale)
        results = controller.test_person(p_id)
        total_accuracy += (sum(results.values()) / len(results)) / 30
    print(total_accuracy)

def main():
    p_id = 3
    model = 'Sarsa'

    # apply_person(model, p_id, 10000, 50, 1)
    apply_all(model, 10000, 50, 2)
    # TODO - for some person: 2, 3, 4 are all wrong BUG ???
    # later same type items is categoried into other bucket
    # TODO - temp solution: 1.set alpha & beta when updating
    #                       2.record the clothes and shuffle them to update
    #                       3.set train time and reward_scale according to TRUE / FALSE

if '__main__' == __name__:
    main()
