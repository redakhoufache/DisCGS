package DPMM
import Common.Tools._
import breeze.linalg.{DenseMatrix, DenseVector, sum}
import breeze.numerics.log
import scala.collection.parallel.CollectionConverters._
import scala.annotation.tailrec
import scala.collection.mutable.ListBuffer

/** Implements the inference of a Dirichlet Process Mixture Model on a multivariate dataset of continuous observations.
 *
 */
class MasterGibbsSampler(workerResults:List[(Int,Int, (DenseVector[Double], DenseMatrix[Double]), Int)],
                 alpha: Double, prior: NormalInverseWishart) extends Serializable {

  val n: Int = workerResults.length

  val workerIds: List[Int] = workerResults.map(e=>e._1)
  val weights: List[Int] = workerResults.map(e=>e._4)
  val means: List[DenseVector[Double]] = workerResults.map(e=>e._3._1)
  val squaredSums: List[DenseMatrix[Double]] =  workerResults.map(e=>e._3._2)
  val Data: List[(DenseVector[Double], DenseMatrix[Double], Int)] = (0 until n ).toList.map(i => (means(i), squaredSums(i), weights(i)))
  val weightsDv: DenseVector[Double] = DenseVector(weights.map(_.toDouble).toArray)
  val sumWeights: Double = sum(weights)
  val d: Int = means(0).length
  //private  var rowPartition: List[Int] = (0 until n).toList
  private  var rowPartition: List[Int] = List.fill(n)(0)
  var partitionEveryIteration: List[List[Int]] = List(rowPartition)

  def aggregateMeans(ms: List[DenseVector[Double]], ws: List[Int]): DenseVector[Double] = {
    val sums = sum((ms zip ws).map(e=>e._2.toDouble * e._1))/sum(ws).toDouble
    sums
  }

  def aggregateSquaredSums(sS: List[DenseMatrix[Double]], ms: List[DenseVector[Double]],ws: List[Int], aggMean: DenseVector[Double]): DenseMatrix[Double] = {
    val aggM: DenseMatrix[Double] = sum(sS) + sum((ms zip ws).map(e=>(e._1 * e._1.t) * e._2.toDouble)) - sum(ws).toDouble * aggMean * aggMean.t
    aggM
  }

  var NIWParams: ListBuffer[NormalInverseWishart] = (Data zip rowPartition).groupBy(_._2).values.map(e => {
    val DataPerCluster: List[(DenseVector[Double], DenseMatrix[Double], Int)] = e.map(_._1)
    val clusterIdx = e.head._2
    val meansPerCluster = DataPerCluster.map(_._1)
    val weightsPerCluster = DataPerCluster.map(_._3)
    val aggregatedMeans: DenseVector[Double] = aggregateMeans(meansPerCluster, weightsPerCluster)
    val squaredSumsPerCluster: List[DenseMatrix[Double]] = DataPerCluster.map(_._2)
    val aggregatedsS: DenseMatrix[Double] = aggregateSquaredSums(sS = squaredSumsPerCluster, ms = meansPerCluster, ws = weightsPerCluster, aggMean = aggregatedMeans )
    (clusterIdx, prior.updateFromSufficientStatistics(weight = sum(weightsPerCluster), mean = aggregatedMeans, SquaredSum = aggregatedsS ))
  }).toList.sortBy(_._1).map(_._2).to(ListBuffer)

  var actualAlpha: Double = alpha

  /** Computes the prior predictive distribution of one observation
   *
   * @param idx Observation index
   */
  def priorPredictive(idx: Int): Double = {
    prior.priorPredictiveFromSufficientStatistics(weights(idx), means(idx), squaredSums(idx))
  }

  /** Computes the cluster membership (existing clusters + new cluster discovery) probabilities.
   *
   * @param idx Index of the target observation
   */
  def computeClusterMembershipProbabilities(idx: Int): List[Double] = {

    NIWParams.indices.par.map(k => {
      (k,
        NIWParams(k).priorPredictiveFromSufficientStatistics(weights(idx), means(idx), squaredSums(idx))
          + log(NIWParams(k).nu - this.d)
      )
    }).toList.sortBy(_._1).map(_._2)
  }

  /** Update the membership of one observation
   *
   * @param idx Index of the target observation
   */
  def drawMembership(idx: Int): Int = {

    val probPartition = computeClusterMembershipProbabilities(idx)
    val posteriorPredictiveXi = priorPredictive(idx)
    val probs = probPartition :+ (posteriorPredictiveXi + log(actualAlpha))
    val normalizedProbs = normalizeLogProbability(probs)
    sample(normalizedProbs)
  }

  /** Before updating one observation's membership, decrements the cluster count and removes the information associated with
   * that observation from the corresponding block components (i.e. all the block components belonging to the associated row-cluster)
   *
   * @param idx Index of the target observation
   * @param formerMembership Previous membership of the target observation
   */

  private def removeElementFromCluster(idx: Int): Unit = {
    val formerMembership: Int =  rowPartition(idx)
    if (NIWParams(formerMembership).nu ==  weights(idx) + d + 1 ) {
      NIWParams.remove(formerMembership)
      rowPartition = rowPartition.map(c => {if(c > rowPartition(idx)){c - 1} else c})
    } else {
      val updatedNIWParams = NIWParams(formerMembership).removeFromSufficientStatistics(weights(idx), means(idx), squaredSums(idx))
      NIWParams.update(formerMembership, updatedNIWParams)
    }
  }

  /** After having sampled a new membership for observation 'idx', update the NIW hyper-parameters corresponding
   * to that cluster with the value of these observations
   *
   * @param idx The index of the observation to use for update
   */

  def addElementToCluster(idx: Int): Unit = {
    val newPartition = rowPartition(idx)
    if (newPartition == NIWParams.length) {
      val newNIWparam = this.prior.updateFromSufficientStatistics(weights(idx), means(idx), squaredSums(idx))
      NIWParams = NIWParams ++ ListBuffer(newNIWparam)
    } else {
      val updatedNIWParams = NIWParams(newPartition).updateFromSufficientStatistics(weights(idx), means(idx), squaredSums(idx))
      NIWParams.update(newPartition, updatedNIWParams)
    }
  }

  /** Update the membership of every observations in the dataset
   *
   */

  def updatePartition(): Unit = {
    for (idx <- means.indices) {
      val currentPartition = rowPartition(idx)
      removeElementFromCluster(idx)
      val newPartition = drawMembership(idx)
      rowPartition = rowPartition.updated(idx, newPartition)
      addElementToCluster(idx)
    }

    partitionEveryIteration = partitionEveryIteration :+ rowPartition
  }

  /** launches the inference process
   *
   * @param nIter Iteration number
   * @param verbose Boolean activating the output of additional information (cluster count evolution)
   * @return
   */
  def run(nIter: Int = 1,
          verbose: Boolean = false): (List[Int], List[NormalInverseWishart]) = {

    @tailrec def go(iter: Int): Unit = {

      if(verbose){
        println("\n Clustering >>>>>> Iteration: " + iter.toString)
        println(NIWParams.map(_.nu - d).toList)
      }
      if (iter <= nIter) {
        var t0 = System.nanoTime()
        updatePartition()
        go(iter + 1)
      }
    }
    go(1)
    (partitionEveryIteration.last, NIWParams.toList)
  }
}
