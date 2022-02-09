import HierarchyAnalysisMatrix as ham

def test_matrix():
    m = ham.HierarchyAnalysisMatrix(features=["Стоимость", 'Стоимость расходников', 'Шум', 'Качество'],
                                    examples=['Матричный', 'Струйный', 'Лазерный'])
    m.set_value_feature('Стоимость расходников', 'Стоимость', 4)
    m.set_value_feature('Шум', 'Стоимость', 2)
    m.set_value_feature('Качество', 'Стоимость', 0.5)
    m.set_value_feature('Шум', 'Стоимость расходников', 1 / 3)
    m.set_value_feature('Качество', 'Стоимость расходников', 1 / 6)
    m.set_value_feature('Качество', 'Шум', 1 / 6)

    f_p = [round(value, 2) for value in m.get_feature_priority_vector()]
    assert f_p == [0.26, 0.06, 0.13, 0.54]

    f_c = [round(value, 2) for value in m.get_feature_concord_data()]
    assert f_c == [4.1, 0.03, 3.71]

    m.set_value_example_by_feature('Стоимость', 'Струйный', 'Матричный', 2)
    m.set_value_example_by_feature('Стоимость', 'Лазерный', 'Матричный', 4)
    m.set_value_example_by_feature('Стоимость', 'Лазерный', 'Струйный', 3)

    m.set_value_example_by_feature('Стоимость расходников', 'Струйный', 'Матричный', 4)
    m.set_value_example_by_feature('Стоимость расходников', 'Лазерный', 'Матричный', 9)
    m.set_value_example_by_feature('Стоимость расходников', 'Лазерный', 'Струйный', 2)

    m.set_value_example_by_feature('Шум', 'Струйный', 'Матричный', 1/6)
    m.set_value_example_by_feature('Шум', 'Лазерный', 'Матричный', 1/8)
    m.set_value_example_by_feature('Шум', 'Лазерный', 'Струйный', 1/2)

    m.set_value_example_by_feature('Качество', 'Струйный', 'Матричный', 1/6)
    m.set_value_example_by_feature('Качество', 'Лазерный', 'Матричный', 1/9)
    m.set_value_example_by_feature('Качество', 'Лазерный', 'Струйный', 1/3)

    res = m.calculate_result_priorities_for_examples()
    assert [round(value, 2)  for value in res["Priority"]] == [0.23, 0.29, 0.47]


