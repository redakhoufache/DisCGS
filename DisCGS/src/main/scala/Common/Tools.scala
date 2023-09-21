package Common
import breeze.linalg.eigSym.EigSym
import breeze.linalg.{DenseMatrix, DenseVector, eigSym, max, sum}
import breeze.numerics.{exp, log}
import breeze.stats.distributions.{Beta, Gamma}

object Tools extends java.io.Serializable {

  def getDatasetInformation(datasetName: String, datasetsHandelInfo: Map[String, Map[String, Any]]): Option[Map[String, Any]] = {
    datasetsHandelInfo.get(datasetName)}


    def getRunningTime(t1: Long, t0: Long): Double = {
    (t1 - t0) / 1e9D
  }

  def strLineToDenseVector(line: String): DenseVector[Double] = {
    val listLine = line.split(" ").toList.map(_.toDouble)
    DenseVector(listLine: _*)
  }

   def convertIteratorToList(data: Iterator[(Int, DenseVector[Double])]): (List[Int], List[DenseVector[Double]]) = {

    var dataList: List[DenseVector[Double]] = List()
    var indexList: List[Int] = List()
    while (data.hasNext) {
      val tup = data.next
      dataList = dataList :+ tup._2
      indexList = indexList :+ tup._1
    }
    (indexList, dataList)
  }

  def covariance(X: List[DenseVector[Double]], mode: DenseVector[Double]): DenseMatrix[Double] = {

    require(mode.length==X.head.length)
    val XMat: DenseMatrix[Double] = DenseMatrix(X.toArray:_*)
    val p = XMat.cols

    val modeMat: DenseMatrix[Double] = DenseMatrix.ones[Double](X.length,1) * mode.t
    val XMatCentered: DenseMatrix[Double] = XMat - modeMat
    val covmat = (XMatCentered.t * XMatCentered)/ (X.length.toDouble-1)

    round(covmat,8)
  }

  def mean(X: List[DenseVector[Double]]): DenseVector[Double] = {
    require(X.nonEmpty)
    X.reduce(_+_) / X.length.toDouble
  }

  def updateAlpha(alpha: Double, alphaPrior: Gamma, nCluster: Int, nObservations: Int): Double = {
    val shape = alphaPrior.shape
    val rate =  1D / alphaPrior.scale

    val log_x = log(new Beta(alpha + 1, nObservations).draw())
    val pi1 = shape + nCluster + 1
    val pi2 = nObservations * (rate - log_x)
    val pi = pi1 / (pi1 + pi2)
    val newScale = 1 / (rate - log_x)

    max(if(sample(List(pi, 1 - pi)) == 0){
      Gamma(shape = shape + nCluster, newScale).draw()
    } else {
      Gamma(shape = shape + nCluster - 1, newScale).draw()
    }, 1e-8)
  }

  def relabel[T: Ordering](L: List[T]): List[Int] = {
    val uniqueLabels = L.distinct.sorted
    val dict = uniqueLabels.zipWithIndex.toMap
    L.map(dict)
  }

  def checkPosDef(M: DenseMatrix[Double]): Unit = {
    val EigSym(lambda, _) = eigSym(M)
    assert(lambda.forall(_>0))
  }


  def normalizeProbability(probs: List[Double]): List[Double] = {
    normalizeLogProbability(probs.map(e => log(e)))
  }

  def logSumExp(X: List[Double]): Double ={
    val maxValue = max(X)
    maxValue + log(sum(X.map(x => exp(x-maxValue))))
  }

  def logSumExp(X: DenseVector[Double]): Double ={
    val maxValue = max(X)
    maxValue + log(sum(X.map(x => exp(x-maxValue))))
  }

  def normalizeLogProbability(probs: List[Double]): List[Double] = {
    val LSE = logSumExp(probs)
    probs.map(e => exp(e - LSE))
  }

  def sample(probabilities: List[Double]): Int = {
    val dist = probabilities.indices zip probabilities
    val threshold = scala.util.Random.nextDouble
    val iterator = dist.iterator
    var accumulator = 0.0
    while (iterator.hasNext) {
      val (cluster, clusterProb) = iterator.next
      accumulator += clusterProb
      if (accumulator >= threshold)
        return cluster
    }
    sys.error("Error")
  }

  def getPartitionFromSize(size: List[Int]): List[Int] = {
    size.indices.map(idx => List.fill(size(idx))(idx)).reduce(_ ++ _)
  }

  def factorial(n: Double): Double = {
    if (n == 0) {1} else {n * factorial(n-1)}
  }

  def logFactorial(n: Double): Double = {
    //if (n == 0) {0} else {log(n) + logFactorial(n-1)}
    var r: Double = 0
    var m: Double = n
    while(m!=0){
      r=r+log(m)
      m=m-1
    }
    r
  }

  // count the number of elements in each cluster of a given partition and return the list, sorted by cluster index
  def partitionToOrderedCount(membership: List[Int]): List[Int] = {
    membership.groupBy(identity).mapValues(_.size).toList.sortBy(_._1).map(_._2)
  }

  def round(m: DenseMatrix[Double], digits:Int): DenseMatrix[Double] = {
    m.map(round(_,digits))
  }

  def round(x: Double, digits:Int): Double = {
    val factor: Double = Math.pow(10,digits)
    Math.round(x*factor)/factor
  }

}