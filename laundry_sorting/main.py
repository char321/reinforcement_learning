from components.Controller import Controller
from components.Config import Config
import numpy as np
import pandas as pd
import pprint
from data_loader.DataLoader import DataLoader


def test_all():
    controller = Controller()

    controller.train()
    controller.test_all()
    print(controller.get_q_table())


def apply_person(p_id):
    controller = Controller(Config())

    controller.train()
    controller.test_person(p_id)
    # print(controller.get_q_table())
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(controller.data[p_id])

    controller.set_user(p_id)
    controller.apply(p_id)
    controller.test_person(p_id)
    # print(controller.get_q_table())


def apply_all(config):
    controller = Controller(config)

    controller.train()
    q_table = np.copy(controller.get_q_table())
    before_total_accuracy = 0
    before_list = []
    for p_id in range(1, 31):
        controller.reload_default_policy()
        results = controller.test_person(p_id)
        before_total_accuracy += (sum(results.values()) / len(results)) / 30
        before_list.append(round(sum(results.values()) / len(results), 3))

    # print(before_total_accuracy)

    after_total_accuracy = 0
    after_list = []
    for p_id in range(1, 31):
        controller.reload_default_policy()
        controller.set_user(p_id)
        controller.apply(p_id)
        results = controller.test_person(p_id)
        after_total_accuracy += (sum(results.values()) / len(results)) / 30
        after_list.append(round(sum(results.values()) / len(results), 3))
        # print(sum(results.values()) / len(results))

    # print(after_list)
    # print(after_total_accuracy)
    return [before_total_accuracy, after_total_accuracy, after_list, min(after_list), max(after_list)]


def tuning():
    config = Config()
    controller = Controller(config)

    for parameter_list in config.combinations:
        print(parameter_list)
        config.set_parameters(parameter_list)
        controller.train()


def tuning2():
    config = Config()
    result = pd.DataFrame(columns=('idx', 'parameter', 'ac1', 'ac2'))
    id = 0
    for parameter_list in config.combinations:
        print('combination ' + str(id + 1))
        print(parameter_list)
        config.set_parameters(parameter_list)

        total_ac1, total_ac2 = 0, 0

        # [ac1, ac2, all_ac, mi, ma] = apply_all(config)

        for i in range(10):
            [ac1, ac2, all_ac, mi, ma] = apply_all(config)
            total_ac1 += ac1
            total_ac2 += ac2

        print(total_ac1 / 10)
        print(total_ac2 / 10)
        data = pd.DataFrame(
            {'idx': id, 'parameter': str(parameter_list), 'ac1': total_ac1 / 10, 'ac2': total_ac2 / 10},
            index=[0])
        result = result.append(data, ignore_index=True)
        id += 1

    csv_filename = 'tuning_result.csv'
    result.to_csv(csv_filename, float_format='%.5f', index=True, header=True)
    print(result)


def main():
    p_id = 1

    test = Config()
    print(test.number_of_combinations)

    # controller = Controller(Config())
    # controller.train()

    # apply_person(p_id)

    # apply_all()

    # tuning()

    tuning2()

    # TODO - for some person: 2, 3, 4 are all wrong BUG ???
    # later same type items is categoried into other bucket
    # TODO - temp solution: 1.set alpha & beta when updating
    #                       2.record the clothes and shuffle them to update
    #                       3.set train time and reward_scale according to TRUE / FALSE


def test():
    # dataLoader = DataLoader()
    # dataLoader.image_aug()

    # res = []
    # acc = []
    # for p_id in range(1, 31):
    #     controller = Controller(Config())
    #     controller.set_user(p_id)
    #     (results, accuracy) = controller.train_with_dqn()
    #     res.append(results)
    #     acc.append(accuracy)
    # # print(res)
    # print(acc)

    # controller = Controller(Config())
    # controller.set_user(1)
    # (results, accuracy) = controller.apply_with_dqn()
    # print(results)
    # print(accuracy)

    controller = Controller(Config())
    controller.train_with_dqn()
    # controller.test_with_dqn()


if '__main__' == __name__:
    # main()
    test()
