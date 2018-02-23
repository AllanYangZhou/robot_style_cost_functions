import tensorflow as tf


def add_simple_summary(summary_writer, tag, value, step):
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, step)
