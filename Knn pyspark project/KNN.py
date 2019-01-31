import pyspark
import heapq
import operator as op

K = 5

def distanceSquared (a, b):
    return sum((x1-x2)**2 for (x1, x2) in zip(a, b))

def simple_csv_loader (line):
    split = line.split(",")
    return (int(split[0]), (float(split[1]), float(split[2])))

# p: (id<p>, sample<p>)
# samples: [(id<s>, (class<s>, sample<s>))]
# return: (id<p>, class<p>)
def kNNWorker (p, samples):
    nsm = heapq.nsmallest(K, samples, key=lambda s: distanceSquared(s[1][1], p[1]))
    c = {}
    for (_, (cls, _)) in nsm:
        c[cls] = c.get(cls, 0) + 1
    best_class = max(c.iteritems(), key=op.itemgetter(1))[0]
    return (p[0], best_class)

# training_rdd: RDD([(id<s>, (class<s>, sample<s>))])
# testing_rdd: RDD([(id<p>, sample<p>)])
# return: RDD([(id<p>, class<p>)])
def kNN (sc, training_rdd, testing_rdd):
    broad_training = sc.broadcast(training_rdd.collect())
    return testing_rdd.map(lambda p: kNNWorker(p, broad_training.value))

# rdd: RDD([(id, (class, sample))])
# return: RDD([(id, (class<true>, class<pred>))])
def testKNN (sc, rdd, split=0.5, seed=None):
    train, test = rdd.randomSplit([split, 1-split], seed)
    hiddenTest = test.mapValues(op.itemgetter(1))
    knnOut = kNN(sc, train, hiddenTest)
    return test.mapValues(op.itemgetter(0)).join(knnOut)

def main (sc, training_set, testing_set, out_path):
    training_data = sc.textFile(training_set).map(simple_csv_loader)
    testing_data = sc.textFile(testing_set).map(simple_csv_loader).values()
    kNN(sc, trainingData, testing_data).saveAsTextFile(out_path)
    return None

if __name__ == "__main__":
    sc = pyspark.SparkContext()
    main(sc)
