import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf, tensorflow_probability as tfp
tfd = tfp.distributions; d = lambda v: tf.constant(v, tf.float64)
print(tfp.__version__, tf.__version__)
c = tfd.Chi2(d(1.80006))
print("Chi2(1.80006).sf(90.5982)", float(c.survival_function(d(90.5982))), "igammac", float(tf.math.igammac(d(1.80006)/2, d(90.5982)/2)), "1-cdf", 1 - float(c.cdf(d(90.5982))))
print("Chi2(20).sf(100)", float(tfd.Chi2(d(20.0)).survival_function(d(100.0))), "igammac", float(tf.math.igammac(d(10.0), d(50.0))))
print("Gamma(10,0.5).sf(100)", float(tfd.Gamma(d(10.0), d(0.5)).survival_function(d(100.0))))
print("Binomial(50,.01).sf(25)", float(tfd.Binomial(total_count=d(50.0), probs=d(0.01)).survival_function(d(25.0))))
print("Binomial(100,.75).sf(99)", float(tfd.Binomial(total_count=d(100.0), probs=d(0.75)).survival_function(d(99.0))))
print("Chi2(1e6).cdf(1e6)", float(tfd.Chi2(d(1e6)).cdf(d(1e6))), "Chi2(1e6).quantile(.5)", float(tfd.Chi2(d(1e6)).quantile(d(0.5))))
print("StudentT(1.5).quantile(1e-200)", float(tfd.StudentT(d(1.5), d(0.0), d(1.0)).quantile(d(1e-200))))
print("StudentT(1).quantile(1e-200)", float(tfd.StudentT(d(1.0), d(0.0), d(1.0)).quantile(d(1e-200))))
print("Beta(50,200).sf(.5)", float(tfd.Beta(d(50.0), d(200.0)).survival_function(d(0.5))), "betainc complement", float(tf.math.betainc(d(200.0), d(50.0), d(0.5))))
print("Beta(1,3).quantile(.5)", float(tfd.Beta(d(1.0), d(3.0)).quantile(d(0.5))), "closed form", 1 - 0.5 ** (1 / 3))
print("Beta(59824,5.20667).quantile(1.0354e-55)", float(tfd.Beta(d(59824.0), d(5.20667)).quantile(d(1.0354e-55))))
print("Beta(3988.55,0.214992).quantile(1.11027e-46)", float(tfd.Beta(d(3988.55), d(0.214992)).quantile(d(1.11027e-46))))
print("StudentT(1e5) f32 cdf(1000)", float(tfd.StudentT(tf.constant(1e5), tf.constant(0.0), tf.constant(1.0)).cdf(tf.constant(1000.0))))
