import numpy as np
import pandas as pd

import tensorflow as tf

data_url = "train.csv"
data = pd.read_csv(data_url) #pandas로 csv 파일을 불러온다.

# first = 0
# second = 0
# third = 0
#
# sur = 0
#
# print(data["Pclass"].value_counts())
# print(data["Sex"].value_counts())

# pclass = data[["Survived", "Pclass"]]
#
# for index, data in pclass.iterrows():
#     if data["Survived"] == 1:
#         if data["Pclass"] == 1:
#             first +=1
#         elif data["Pclass"] == 2:
#             second +=1
#         else:
#             third +=1
#
# print("data : ", first, second, third)





# sex = data[["Survived", "Sex"]]
#
# male = 0
# female = 0
#
# for index, data in sex.iterrows():
#     if data["Survived"] == 1:
#         if data["Sex"] == "male":
#             male +=1
#         else:
#             female +=1
#
# print("sex : ", male, female)

#등급 :: 216, 184, 491 : 생존자 136, 87, 119
#성별 :: male      577, female    314 : 생존자 : 109, 233

#등급 : 0.63, 0.47, 0.24
#성별 : 0.19, 0.74



######################################################
x = data[["Pclass", "Sex"]]
y = data["Survived"].copy()

x["Pclass"] = x["Pclass"].astype(float)

print(x.size)








for index, data in x.iterrows():
    if data["Pclass"] == 1:
        # x[index]["Pclass] = 0.63
        x.set_value(index, 'Pclass', 0.63)
    elif data["Pclass"] == 2:
        # x[index]["Pclass"] = 0.47
        x.set_value(index, 'Pclass', 0.47)
    else:
        # x[index]["Pclass"] = 0.24
        x.set_value(index, 'Pclass', 0.24)

    if data["Sex"] == "male":
        # x[index]["Sex"] = 0.19
        x.set_value(index, 'Sex', 0.19)
    else:
        # x[index]["Sex"] = 0.74
        x.set_value(index, 'Sex', 0.74)

# print("asd", x.as_matrix().shape)

x = x.as_matrix()
y = y.as_matrix()
y = y.reshape((-1, 1))





test = pd.read_csv("test.csv")

x_data = test[["Pclass", "Sex"]]

x_data["Pclass"] = x_data["Pclass"].astype(float)

for index, data in x_data.iterrows():
    if data["Pclass"] == 1:
        # x[index]["Pclass] = 0.63
        x_data.set_value(index, 'Pclass', 0.63)
    elif data["Pclass"] == 2:
        # x[index]["Pclass"] = 0.47
        x_data.set_value(index, 'Pclass', 0.47)
    else:
        # x[index]["Pclass"] = 0.24
        x_data.set_value(index, 'Pclass', 0.24)

    if data["Sex"] == "male":
        # x[index]["Sex"] = 0.19
        x_data.set_value(index, 'Sex', 0.19)
    else:
        # x[index]["Sex"] = 0.74
        x_data.set_value(index, 'Sex', 0.74)

# print("asd", x.as_matrix().shape)

x_data = x_data.as_matrix()










# print("x : ", x.shape)
# print("y : ", y.shape)
#
X = tf.placeholder("float", [None, 2]) #shape x 데이터는 n개(None), 각 요소는 2개의 값을 가진다.
Y = tf.placeholder("float", [None, 1]) #shape y 데이터는 n개(None), 각 요소는 1개의 값을 가진다.

W = tf.Variable(tf.random_normal([2, 1]), name="weight") #가중치의 shape는 들어오는 값 input(2), 나가는 값 output(1)
b = tf.Variable(tf.random_normal([1]), name="bias") #편향은 출력값과 같다.

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)






# define cost/loss & optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
# optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
#
# prediction = tf.argmax(hypothesis, 1) #예측된 값에서 가장 높은 값을 가지는 요소의 인덱스 추출
# correct_prediction = tf.equal(prediction, tf.argmax(Y, 1)) #정답레이블과 예상치 비교
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #정확도 계산




# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) #최적화함수, 학습률
train = optimizer.minimize(cost) #

predicted = tf.cast(hypothesis > 0.35, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))









# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #초기화

    for step in range(10):
        a = sess.run(accuracy, feed_dict={X: x, Y: y})
        # print("Accuracy: {}".format(a))

    # for step in range(890):
    #     # sess.run(train, feed_dict={X: x, Y: y}) #optimizer에 모든 텐서가 연결되어 있으므로 optimizer 실행
    #     # loss, acc = sess.run([cost, accuracy], feed_dict={X: x, Y: y})
    #     # print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))
    #     h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x, Y: y})
    #     print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)







    # Let's see if we can predict
    # 학습 이후 테스트
    # pred = sess.run(prediction, feed_dict={X: x_data})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    # for p, y in zip(pred, y_data.flatten()): #flattend은 1차원으로 변환. [[1], [0]] -> [1, 0].
    #     #zip으로 묶어 각 값을 p와 y로
    #     print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))


    pred = sess.run(predicted, feed_dict={X: x_data})
    print(pred)



    # for step in range(5):
    #     pred = sess.run(predicted, feed_dict={X: x_data})
    #     print(pred)
