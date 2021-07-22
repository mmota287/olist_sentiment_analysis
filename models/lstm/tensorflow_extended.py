from typing import Any
import tensorflow as tf

class ExtendedTensorBoard(tf.keras.callbacks.TensorBoard):
  """It's an extension of the [tf.keras.callbacks.TensorBoard] that writes training information

  Args:
      tf ([type]): [description]
  """

  def __init__(self, test_dataset: tf.data.Dataset, logs_dir: str, histogram_freq: int = 1):
    
    super(ExtendedTensorBoard, self).__init__(log_dir = logs_dir, histogram_freq = histogram_freq)
    self.test_dataset = test_dataset
  
  def _log_gradients(self, epoch: int):
    """ Private method for writing weight gradients

    Args:
        epoch (int): [description]
    """

    writer = tf.summary.create_file_writer(self.log_dir)

    with writer.as_default(), tf.GradientTape() as g:
      features, y_true = list(self.test_dataset.batch(100).take(1))[0]
      weights = [w for w in self.model.trainable_weights if 'dense' in w.name or 'bias' in w.name or 'lstm' in w.name ]
      features, y_true = list(self.test_dataset.batch(100).take(1))[0]
      y_pred = self.model(features)
      loss = self.model.compiled_loss(y_true= y_true , y_pred=y_pred)
      gradients = g.gradient(loss, weights)
      
      for w, grads in zip(weights, gradients):
        curr_grad = grads[0]
        mean = tf.reduce_mean(tf.abs(curr_grad))

        tf.summary.scalar(f'grad_mean_weight_{w.name}', mean, step=epoch)
        tf.summary.histogram(f'grad_histogram_weight_{w.name}', curr_grad, step=epoch)
          
  def on_epoch_end(self, epoch: int, logs:Any = None):
    """Method called when epoch ends

    Args:
        epoch (int): Training epoch
        logs (Any, optional): Training logs. Defaults to None.
    """


    super(ExtendedTensorBoard, self).on_epoch_end(epoch, logs=logs)

    if self.histogram_freq and epoch % self.histogram_freq == 0:
      self._log_gradients(epoch)
          