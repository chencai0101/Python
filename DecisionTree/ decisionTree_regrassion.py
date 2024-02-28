import graphviz as gv
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.metrics import accuracy_score

def decision_tree_example():
    # 加载鸢尾花数据集
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names

    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建决策树分类器
    clf = DecisionTreeRegressor()

    # 在训练集上训练决策树
    clf.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = clf.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # 绘制决策树
    dot_data = export_graphviz(clf, out_file=None, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, special_characters=True, node_ids=True)
    graph = gv.Source(dot_data)
    graph.render(filename='iris_decision_tree')

    # 显示图像
    gv.view('iris_decision_tree.pdf')

if __name__ == "__main__":
    decision_tree_example()