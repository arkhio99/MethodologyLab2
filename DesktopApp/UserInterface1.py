# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '..\..\DesktopApp\form.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1749, 731)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.table_feature = QtWidgets.QTableWidget(self.centralwidget)
        self.table_feature.setGeometry(QtCore.QRect(10, 150, 681, 261))
        self.table_feature.setObjectName("table_feature")
        self.table_feature.setColumnCount(0)
        self.table_feature.setRowCount(0)
        self.btn_add_feature = QtWidgets.QPushButton(self.centralwidget)
        self.btn_add_feature.setGeometry(QtCore.QRect(540, 46, 93, 28))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.btn_add_feature.setFont(font)
        self.btn_add_feature.setObjectName("btn_add_feature")
        self.text_editor_feature = QtWidgets.QLineEdit(self.centralwidget)
        self.text_editor_feature.setGeometry(QtCore.QRect(10, 48, 531, 27))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.text_editor_feature.setFont(font)
        self.text_editor_feature.setObjectName("text_editor_feature")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 15, 171, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.btn_add_example = QtWidgets.QPushButton(self.centralwidget)
        self.btn_add_example.setGeometry(QtCore.QRect(1370, 38, 91, 28))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.btn_add_example.setFont(font)
        self.btn_add_example.setObjectName("btn_add_example")
        self.text_editor_example = QtWidgets.QLineEdit(self.centralwidget)
        self.text_editor_example.setGeometry(QtCore.QRect(892, 38, 481, 27))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.text_editor_example.setFont(font)
        self.text_editor_example.setObjectName("text_editor_example")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(900, 4, 191, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.table_example_by_feature = QtWidgets.QTableWidget(self.centralwidget)
        self.table_example_by_feature.setGeometry(QtCore.QRect(890, 170, 631, 241))
        self.table_example_by_feature.setObjectName("table_example_by_feature")
        self.table_example_by_feature.setColumnCount(0)
        self.table_example_by_feature.setRowCount(0)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 120, 191, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.list_features = QtWidgets.QComboBox(self.centralwidget)
        self.list_features.setGeometry(QtCore.QRect(890, 140, 351, 22))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.list_features.setFont(font)
        self.list_features.setObjectName("list_features")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(890, 110, 381, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(20, 420, 211, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(177, 440, 41, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(176, 460, 41, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.lbl_eigen_value_feature = QtWidgets.QLabel(self.centralwidget)
        self.lbl_eigen_value_feature.setGeometry(QtCore.QRect(220, 420, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lbl_eigen_value_feature.setFont(font)
        self.lbl_eigen_value_feature.setText("")
        self.lbl_eigen_value_feature.setObjectName("lbl_eigen_value_feature")
        self.lbl_concord_index_feature = QtWidgets.QLabel(self.centralwidget)
        self.lbl_concord_index_feature.setGeometry(QtCore.QRect(220, 438, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lbl_concord_index_feature.setFont(font)
        self.lbl_concord_index_feature.setText("")
        self.lbl_concord_index_feature.setObjectName("lbl_concord_index_feature")
        self.lbl_concord_estimate_feature = QtWidgets.QLabel(self.centralwidget)
        self.lbl_concord_estimate_feature.setGeometry(QtCore.QRect(220, 457, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lbl_concord_estimate_feature.setFont(font)
        self.lbl_concord_estimate_feature.setText("")
        self.lbl_concord_estimate_feature.setObjectName("lbl_concord_estimate_feature")
        self.lbl_eigen_value_example = QtWidgets.QLabel(self.centralwidget)
        self.lbl_eigen_value_example.setGeometry(QtCore.QRect(1364, 420, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lbl_eigen_value_example.setFont(font)
        self.lbl_eigen_value_example.setText("")
        self.lbl_eigen_value_example.setObjectName("lbl_eigen_value_example")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(1160, 420, 201, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.lbl_concord_estimate_example = QtWidgets.QLabel(self.centralwidget)
        self.lbl_concord_estimate_example.setGeometry(QtCore.QRect(1364, 457, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lbl_concord_estimate_example.setFont(font)
        self.lbl_concord_estimate_example.setText("")
        self.lbl_concord_estimate_example.setObjectName("lbl_concord_estimate_example")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(1321, 440, 41, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.lbl_concord_index_example = QtWidgets.QLabel(self.centralwidget)
        self.lbl_concord_index_example.setGeometry(QtCore.QRect(1364, 438, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lbl_concord_index_example.setFont(font)
        self.lbl_concord_index_example.setText("")
        self.lbl_concord_index_example.setObjectName("lbl_concord_index_example")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(1320, 460, 41, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.btn_calc_feature_priority = QtWidgets.QPushButton(self.centralwidget)
        self.btn_calc_feature_priority.setGeometry(QtCore.QRect(700, 120, 181, 28))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.btn_calc_feature_priority.setFont(font)
        self.btn_calc_feature_priority.setObjectName("btn_calc_feature_priority")
        self.btn_calc_example_priority_by_feature = QtWidgets.QPushButton(self.centralwidget)
        self.btn_calc_example_priority_by_feature.setGeometry(QtCore.QRect(1530, 140, 201, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.btn_calc_example_priority_by_feature.setFont(font)
        self.btn_calc_example_priority_by_feature.setObjectName("btn_calc_example_priority_by_feature")
        self.btn_calc_example_priotity = QtWidgets.QPushButton(self.centralwidget)
        self.btn_calc_example_priotity.setGeometry(QtCore.QRect(860, 490, 311, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.btn_calc_example_priotity.setFont(font)
        self.btn_calc_example_priotity.setObjectName("btn_calc_example_priotity")
        self.cb_feature_1 = QtWidgets.QComboBox(self.centralwidget)
        self.cb_feature_1.setGeometry(QtCore.QRect(10, 80, 241, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.cb_feature_1.setFont(font)
        self.cb_feature_1.setObjectName("cb_feature_1")
        self.cb_feature_2 = QtWidgets.QComboBox(self.centralwidget)
        self.cb_feature_2.setGeometry(QtCore.QRect(250, 80, 211, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.cb_feature_2.setFont(font)
        self.cb_feature_2.setObjectName("cb_feature_2")
        self.text_editor_feature_priority = QtWidgets.QLineEdit(self.centralwidget)
        self.text_editor_feature_priority.setGeometry(QtCore.QRect(460, 80, 81, 31))
        self.text_editor_feature_priority.setObjectName("text_editor_feature_priority")
        self.btn_set_feature_priority = QtWidgets.QPushButton(self.centralwidget)
        self.btn_set_feature_priority.setGeometry(QtCore.QRect(540, 79, 93, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.btn_set_feature_priority.setFont(font)
        self.btn_set_feature_priority.setObjectName("btn_set_feature_priority")
        self.cb_example_1 = QtWidgets.QComboBox(self.centralwidget)
        self.cb_example_1.setGeometry(QtCore.QRect(893, 70, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.cb_example_1.setFont(font)
        self.cb_example_1.setObjectName("cb_example_1")
        self.text_editor_example_priority = QtWidgets.QLineEdit(self.centralwidget)
        self.text_editor_example_priority.setGeometry(QtCore.QRect(1280, 70, 91, 31))
        self.text_editor_example_priority.setObjectName("text_editor_example_priority")
        self.cb_example_2 = QtWidgets.QComboBox(self.centralwidget)
        self.cb_example_2.setGeometry(QtCore.QRect(1080, 70, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.cb_example_2.setFont(font)
        self.cb_example_2.setObjectName("cb_example_2")
        self.btn_set_example_priority = QtWidgets.QPushButton(self.centralwidget)
        self.btn_set_example_priority.setGeometry(QtCore.QRect(1370, 70, 93, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.btn_set_example_priority.setFont(font)
        self.btn_set_example_priority.setObjectName("btn_set_example_priority")
        self.table_feature_priorities = QtWidgets.QTableWidget(self.centralwidget)
        self.table_feature_priorities.setGeometry(QtCore.QRect(800, 150, 81, 261))
        self.table_feature_priorities.setObjectName("table_feature_priorities")
        self.table_feature_priorities.setColumnCount(0)
        self.table_feature_priorities.setRowCount(0)
        self.table_example_priorities_by_feature = QtWidgets.QTableWidget(self.centralwidget)
        self.table_example_priorities_by_feature.setGeometry(QtCore.QRect(1637, 170, 91, 241))
        self.table_example_priorities_by_feature.setObjectName("table_example_priorities_by_feature")
        self.table_example_priorities_by_feature.setColumnCount(0)
        self.table_example_priorities_by_feature.setRowCount(0)
        self.table_example_priorities = QtWidgets.QTableWidget(self.centralwidget)
        self.table_example_priorities.setGeometry(QtCore.QRect(350, 530, 1251, 192))
        self.table_example_priorities.setObjectName("table_example_priorities")
        self.table_example_priorities.setColumnCount(0)
        self.table_example_priorities.setRowCount(0)
        self.btn_load = QtWidgets.QPushButton(self.centralwidget)
        self.btn_load.setGeometry(QtCore.QRect(210, 10, 141, 28))
        self.btn_load.setObjectName("btn_load")
        self.table_feature_eigen_vector = QtWidgets.QTableWidget(self.centralwidget)
        self.table_feature_eigen_vector.setGeometry(QtCore.QRect(700, 150, 91, 261))
        self.table_feature_eigen_vector.setObjectName("table_feature_eigen_vector")
        self.table_feature_eigen_vector.setColumnCount(0)
        self.table_feature_eigen_vector.setRowCount(0)
        self.table_example_eigen_vector_by_feature = QtWidgets.QTableWidget(self.centralwidget)
        self.table_example_eigen_vector_by_feature.setGeometry(QtCore.QRect(1530, 170, 101, 241))
        self.table_example_eigen_vector_by_feature.setObjectName("table_example_eigen_vector_by_feature")
        self.table_example_eigen_vector_by_feature.setColumnCount(0)
        self.table_example_eigen_vector_by_feature.setRowCount(0)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "HierarchyAnalysisMatrix"))
        self.btn_add_feature.setText(_translate("MainWindow", "Добавить"))
        self.label.setText(_translate("MainWindow", "Добавить критерий"))
        self.btn_add_example.setText(_translate("MainWindow", "Добавить"))
        self.label_2.setText(_translate("MainWindow", "Добавить претендента"))
        self.label_3.setText(_translate("MainWindow", "Матрица критериев"))
        self.label_4.setText(_translate("MainWindow", "Матрица претенденов относительно критериев"))
        self.label_5.setText(_translate("MainWindow", "Собственное значение ="))
        self.label_6.setText(_translate("MainWindow", "ИС ="))
        self.label_7.setText(_translate("MainWindow", "ОС ="))
        self.label_8.setText(_translate("MainWindow", "Собственное значение ="))
        self.label_9.setText(_translate("MainWindow", "ИС ="))
        self.label_10.setText(_translate("MainWindow", "ОС ="))
        self.btn_calc_feature_priority.setText(_translate("MainWindow", "Рассчитать"))
        self.btn_calc_example_priority_by_feature.setText(_translate("MainWindow", "Рассчитать"))
        self.btn_calc_example_priotity.setText(_translate("MainWindow", "Рассчитать приоритеты для претендентов"))
        self.btn_set_feature_priority.setText(_translate("MainWindow", "Обновить"))
        self.btn_set_example_priority.setText(_translate("MainWindow", "Обновить"))
        self.btn_load.setText(_translate("MainWindow", "Загрузить пример"))