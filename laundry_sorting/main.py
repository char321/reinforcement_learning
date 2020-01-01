from components.Controller import Controller

def main():
    controller = Controller()
    controller.train()
    print(controller.model.get_q_table())
    controller.test_person(30)
    controller.set_user(30)
    controller.apply(30)
    controller.test_person(30)
    print(controller.model.get_q_table())

    # q_table = np.copy(controller.model.get_q_table())
    # total_accuracy = 0
    # for p_id in range(1, 31):
    #     controller.model.set_q_table(np.copy(q_table))
    #     controller.update(p_id)
    #     results = controller.test_person(p_id)
    #     total_accuracy += (sum(results.values()) / 16) / 30
    # print(total_accuracy)

if '__main__' == __name__:
    main()
