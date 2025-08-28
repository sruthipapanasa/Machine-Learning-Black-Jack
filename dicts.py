hard_hands = {
    ('9','2'):'hit', ('9','3'):'double down', ('9','4'):'double down', ('9','5'):'double down', ('9','6'):'double down', ('9','7'):'hit', ('9','8'):'hit', ('9','9'):'hit', ('9','Ten'):'hit', ('9','Ace'):'hit',
    
    ('10','2'):'double down', ('10','3'):'double down', ('10','4'):'double down', ('10','5'):'double down', ('10','6'):'double down', ('10','7'):'double down', ('10','8'):'double down', ('10','9'):'double down', ('10','Ten'):'hit', ('10','Ace'):'hit',
    
    ('11','2'):'double down', ('11','3'):'double down', ('11','4'):'double down', ('11','5'):'double down', ('11','6'):'double down', ('11','7'):'double down', ('11','8'):'double down', ('11','9'):'double down', ('11','Ten'):'double down', ('11','Ace'):'double down',
    
    ('12','2'):'hit', ('12','3'):'hit', ('12','4'):'stand', ('12','5'):'stand', ('12','6'):'stand', ('12','7'):'hit', ('12','8'):'hit', ('12','9'):'hit', ('12','Ten'):'hit', ('12','Ace'):'hit',
    
    ('13','2'):'stand', ('13','3'):'stand', ('13','4'):'stand', ('13','5'):'stand', ('13','6'):'stand', ('13','7'):'hit', ('13','8'):'hit', ('13','9'):'hit', ('13','Ten'):'hit', ('13','Ace'):'hit',
    
    ('14','2'):'stand', ('14','3'):'stand', ('14','4'):'stand', ('14','5'):'stand', ('14','6'):'stand', ('14','7'):'hit', ('14','8'):'hit', ('14','9'):'hit', ('14','Ten'):'hit', ('14','Ace'):'hit',
    
    ('15','2'):'stand', ('15','3'):'stand', ('15','4'):'stand', ('15','5'):'stand', ('15','6'):'stand', ('15','7'):'hit', ('15','8'):'hit', ('15','9'):'hit', ('15','Ten'):'hit', ('15','Ace'):'hit',
    
    ('16','2'):'stand', ('16','3'):'stand', ('16','4'):'stand', ('16','5'):'stand', ('16','6'):'stand', ('16','7'):'hit', ('16','8'):'hit', ('16','9'):'hit', ('16','Ten'):'hit', ('16','Ace'):'hit',

    ('20', '2'):'stand',
    ('20', '3'):'stand',
    ('20', '4'):'stand',
    ('20', '5'):'stand',
    ('20', '6'):'stand',
    ('20', '7'):'stand',
    ('20', '8'):'stand',
    ('20', '9'):'stand',
    ('20', 'Ten'):'stand',
    ('20', 'Ace'):'stand',
    #
    ('19', '2'):'stand',
    ('19', '3'):'stand',
    ('19', '4'):'stand',
    ('19', '5'):'stand',
    ('19', '6'):'stand',
    ('19', '7'):'stand',
    ('19', '8'):'stand',
    ('19', '9'):'stand',
    ('19', 'Ten'):'stand',
    ('19', 'Ace'):'stand',
    #
    ('18', '2'):'stand',
    ('18', '3'):'stand',
    ('18', '4'):'stand',
    ('18', '5'):'stand',
    ('18', '6'):'stand',
    ('18', '7'):'stand',
    ('18', '8'):'stand',
    ('18', '9'):'stand',
    ('18', 'Ten'):'stand',
    ('18', 'Ace'):'stand',
    #
    ('17', '2'):'stand',
    ('17', '3'):'stand',
    ('17', '4'):'stand',
    ('17', '5'):'stand',
    ('17', '6'):'stand',
    ('17', '7'):'stand',
    ('17', '8'):'stand',
    ('17', '9'):'stand',
    ('17', 'Ten'):'stand',
    ('17', 'Ace'):'stand',
    #
    ('8', '2'):'hit',
    ('8', '3'):'hit',
    ('8', '4'):'hit',
    ('8', '5'):'hit',
    ('8', '6'):'hit',
    ('8', '7'):'hit',
    ('8', '8'):'hit',
    ('8', '9'):'hit',
    ('8', 'Ten'):'hit',
    ('8', 'Ace'):'hit',
    #
    ('7', '2'):'hit',
    ('7', '3'):'hit',
    ('7', '4'):'hit',
    ('7', '5'):'hit',
    ('7', '6'):'hit',
    ('7', '7'):'hit',
    ('7', '8'):'hit',
    ('7', '9'):'hit',
    ('7', 'Ten'):'hit',
    ('7', 'Ace'):'hit',
    #
    ('6', '2'):'hit',
    ('6', '3'):'hit',
    ('6', '4'):'hit',
    ('6', '5'):'hit',
    ('6', '6'):'hit',
    ('6', '7'):'hit',
    ('6', '8'):'hit',
    ('6', '9'):'hit',
    ('6', 'Ten'):'hit',
    ('6', 'Ace'):'hit',
    #
    ('5', '2'):'hit',
    ('5', '3'):'hit',
    ('5', '4'):'hit',
    ('5', '5'):'hit',
    ('5', '6'):'hit',
    ('5', '7'):'hit',
    ('5', '8'):'hit',
    ('5', '9'):'hit',
    ('5', 'Ten'):'hit',
    ('5', 'Ace'):'hit'
}
sph_dict = {
    #A-9
    (('Ace', '9'), '2'):'stand',
    (('Ace', '9'), '3'):'stand',
    (('Ace', '9'), '4'):'stand',
    (('Ace', '9'), '5'):'stand',
    (('Ace', '9'), '6'):'stand',
    (('Ace', '9'), '7'):'stand',
    (('Ace', '9'), '8'):'stand',
    (('Ace', '9'), '9'):'stand',
    (('Ace', '9'), 'Ten'):'stand',
    (('Ace', '9'), 'Ace'):'stand',
    #A-8
    (('Ace', '8'), '2'):'stand',
    (('Ace', '8'), '3'):'stand',
    (('Ace', '8'), '4'):'stand',
    (('Ace', '8'), '5'):'stand',
    (('Ace', '8'), '6'):'double down',
    (('Ace', '8'), '7'):'stand',
    (('Ace', '8'), '8'):'stand',
    (('Ace', '8'), '9') :'stand',
    (('Ace', '8'), 'Ten'):'stand',
    (('Ace', '8'), 'Ace'):'stand',
    #A-7
    (('Ace', '7'), '2'):'double down',
    (('Ace', '7'), '3'):'double down',
    (('Ace', '7'), '4'):'double down',
    (('Ace', '7'), '5'):'double down',
    (('Ace', '7'), '6'):'double down',
    (('Ace', '7'), '7'):'stand',
    (('Ace', '7'), '8'):'stand',
    (('Ace', '7'), '9'):'hit',
    (('Ace', '7'), 'Ten'):'hit',
    (('Ace', '7'), 'Ace'):'hit',
    #A-6
    (('Ace', '6'), '2'):'hit',
    (('Ace', '6'), '3'):'double down',
    (('Ace', '6'), '4'):'double down',
    (('Ace', '6'), '5'):'double down',
    (('Ace', '6'), '6'):'double down',
    (('Ace', '6'), '7'):'hit',
    (('Ace', '6'), '8'):'hit',
    (('Ace', '6'), '9'):'hit',
    (('Ace', '6'), 'Ten'):'hit',
    (('Ace', '6'), 'Ace'):'hit',
    #A-5
    (('Ace', '5'), '2'):'hit',
    (('Ace', '5'), '3'):'hit',
    (('Ace', '5'), '4'):'double down',
    (('Ace', '5'), '5'):'double down',
    (('Ace', '5'), '6'):'double down',
    (('Ace', '5'), '7'):'hit',
    (('Ace', '5'), '8'):'hit',
    (('Ace', '5'), '9'):'hit',
    (('Ace', '5'), 'Ten'):'hit',
    (('Ace', '5'), 'Ace'):'hit',
    #A-4
    (('Ace', '4'), '2'):'hit',
    (('Ace', '4'), '3'):'hit',
    (('Ace', '4'), '4'):'double down',
    (('Ace', '4'), '5'):'double down',
    (('Ace', '4'), '6'):'double down',
    (('Ace', '4'), '7'):'hit',
    (('Ace', '4'), '8'):'hit',
    (('Ace', '4'), '9'):'hit',
    (('Ace', '4'), 'Ten'):'hit',
    (('Ace', '4'), 'Ace'):'hit',
    #A-3
    (('Ace', '3'), '2'):'hit',
    (('Ace', '3'), '3'):'hit',
    (('Ace', '3'), '4'):'hit',
    (('Ace', '3'), '5'):'double down',
    (('Ace', '3'), '6'):'double down',
    (('Ace', '3'), '7'):'hit',
    (('Ace', '3'), '8'):'hit',
    (('Ace', '3'), '9'):'hit',
    (('Ace', '3'), 'Ten'):'hit',
    (('Ace', '3'), 'Ace'):'hit',
    #A-2
    (('Ace', '2'), '2'):'hit',
    (('Ace', '2'), '3'):'hit',
    (('Ace', '2'), '4'):'hit',
    (('Ace', '2'), '5'):'double down',
    (('Ace', '2'), '6'):'double down',
    (('Ace', '2'), '7'):'hit',
    (('Ace', '2'), '8'):'hit',
    (('Ace', '2'), '9'):'hit',
    (('Ace', '2'), 'Ten'):'hit',
    (('Ace', '2'), 'Ace'):'hit',
    #A-A
    (('Ace', 'Ace'), '2'):'split',
    (('Ace', 'Ace'), '3'):'split',
    (('Ace', 'Ace'), '4'):'split',
    (('Ace', 'Ace'), '5'):'split',
    (('Ace', 'Ace'), '6'):'split',
    (('Ace', 'Ace'), '7'):'split',
    (('Ace', 'Ace'), '8'):'split',
    (('Ace', 'Ace'), '9'):'split',
    (('Ace', 'Ace'), 'Ten'):'split',
    (('Ace', 'Ace'), 'Ace'):'split',
    #Ten-Ten
    (('Ten', 'Ten'), '2'):'stand',
    (('Ten', 'Ten'), '3'):'stand',
    (('Ten', 'Ten'), '4'):'stand',
    (('Ten', 'Ten'), '5'):'stand',
    (('Ten', 'Ten'), '6'):'stand',
    (('Ten', 'Ten'), '7'):'stand',
    (('Ten', 'Ten'), '8'):'stand',
    (('Ten', 'Ten'), '9'):'stand',
    (('Ten', 'Ten'), 'Ten'):'stand',
    (('Ten', 'Ten'), 'Ace'):'stand',
    #9-9
    (('9', '9'), '2'):'split',
    (('9', '9'), '3'):'split',
    (('9', '9'), '4'):'split',
    (('9', '9'), '5'):'split',
    (('9', '9'), '6'):'split',
    (('9', '9'), '7'):'stand',
    (('9', '9'), '8'):'split',
    (('9', '9'), '9'):'split',
    (('9', '9'), 'Ten'):'stand',
    (('9', '9'), 'Ace'):'stand',
    #8-8
    (('8', '8'), '2'):'split',
    (('8', '8'), '3'):'split',
    (('8', '8'), '4'):'split',
    (('8', '8'), '5'):'split',
    (('8', '8'), '6'):'split',
    (('8', '8'), '7'):'split',
    (('8', '8'), '8'):'split',
    (('8', '8'), '9'):'split',
    (('8', '8'), 'Ten'):'split',
    (('8', '8'), 'Ace'):'split',
    #7-7
    (('7', '7'), '2'):'split',
    (('7', '7'), '3'):'split',
    (('7', '7'), '4'):'split',
    (('7', '7'), '5'):'split',
    (('7', '7'), '6'):'split',
    (('7', '7'), '7'):'split',
    (('7', '7'), '8'):'hit',
    (('7', '7'), '9'):'hit',
    (('7', '7'), 'Ten'):'hit',
    (('7', '7'), 'Ace'):'hit',
    #6-6
    (('6', '6'), '2'):'split',
    (('6', '6'), '3'):'split',
    (('6', '6'), '4'):'split',
    (('6', '6'), '5'):'split',
    (('6', '6'), '6'):'split',
    (('6', '6'), '7'):'hit',
    (('6', '6'), '8'):'hit',
    (('6', '6'), '9'):'hit',
    (('6', '6'), 'Ten'):'hit',
    (('6', '6'), 'Ace'):'hit',
    #5-5
    (('5', '5'), '2'):'double down',
    (('5', '5'), '3'):'double down',
    (('5', '5'), '4'):'double down',
    (('5', '5'), '5'):'double down',
    (('5', '5'), '6'):'double down',
    (('5', '5'), '7'):'double down',
    (('5', '5'), '8'):'double down',
    (('5', '5'), '9'):'double down',
    (('5', '5'), 'Ten'):'hit',
    (('5', '5'), 'Ace'):'hit',
    #4-4
    (('4', '4'), '2'):'hit',
    (('4', '4'), '3'):'hit',
    (('4', '4'), '4'):'hit',
    (('4', '4'), '5'):'split',
    (('4', '4'), '6'):'split',
    (('4', '4'), '7'):'hit',
    (('4', '4'), '8'):'hit',
    (('4', '4'), '9'):'hit',
    (('4', '4'), 'Ten'):'hit',
    (('4', '4'), 'Ace'):'hit',
    #3-3
    (('3', '3'), '2'):'split',
    (('3', '3'), '3'):'split',
    (('3', '3'), '4'):'split',
    (('3', '3'), '5'):'split',
    (('3', '3'), '6'):'split',
    (('3', '3'), '7'):'split',
    (('3', '3'), '8'):'hit',
    (('3', '3'), '9'):'hit',
    (('3', '3'), 'Ten'):'hit',
    (('3', '3'), 'Ace'):'hit',
    #2-2
    (('2', '2'), '2'):'split',
    (('2', '2'), '3'):'split',
    (('2', '2'), '4'):'split',
    (('2', '2'), '5'):'split',
    (('2', '2'), '6'):'split',
    (('2', '2'), '7'):'split',
    (('2', '2'), '8'):'hit',
    (('2', '2'), '9'):'hit',
    (('2', '2'), 'Ten'):'hit',
    (('2', '2'), 'Ace'):'hit',
    
}









'''
predictions = model(features)
predictions[:5]
tf.nn.softmax(predictions[:5])

#print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
#print("    Labels: {}".format(labels))


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)
  return loss_object(y_true=y, y_pred=y_)

l = loss(model, features, labels, training=False)
print("Loss test: {}".format(l))

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

loss_value, grads = grad(model, features, labels)

print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                          loss(model, features, labels, training=True).numpy()))

## Note: Rerunning this cell uses the same model variables

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 101

for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Track progress
    epoch_loss_avg.update_state(loss_value)  # Add current batch loss
    # Compare predicted label to actual label
    # training=True is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    epoch_accuracy.update_state(y, model(x, training=True))

  # End epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 20 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

model.save('./blackjackmodel')
'''


'''


test_blackjack = "test_blackjack.csv"
batch_size = 30
test_dataset = tf.data.experimental.make_csv_dataset(
    test_blackjack,
    batch_size,
    column_names=column_names,
    label_name='winloss',
    num_epochs=1,
    shuffle=False)
test_dataset = test_dataset.map(pack_features_vector)
test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
  # training=False is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  logits = model(x, training=False)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
tf.stack([y,prediction],axis=1)

#predicting
predict_dataset = tf.convert_to_tensor([
    [2,10,0,1,10,23,10],
    [2,10,0,1,10,23,10],
    [5,6,8,0,0,19,10],
    [9,10,0,0,0,19,10],
    [10,8,0,0,0,18,10],
    [5,9,0,10,0,24,10],
    [9,10,0,0,0,19,10],
    [11,10,0,0,0,21,10],
    [10,4,0,7,0,21,10],
    [2,2,4,3,10,21,9],
])

#example 9 should have put out win, the rest are losses

predictions = model(predict_dataset, training=False)


for i, logits in enumerate(predictions):
    class_idx = tf.argmax(logits).numpy()
    p = tf.nn.softmax(logits)[class_idx]
    name = [class_idx], class_names[class_idx]
    print("Example {} prediction: {} ({:4.1f}%)".format(i,name, 100*p))
'''
