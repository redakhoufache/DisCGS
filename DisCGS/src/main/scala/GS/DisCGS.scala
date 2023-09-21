package GS
import breeze.linalg.{DenseMatrix, DenseVector}
import Common.Tools.{convertIteratorToList, partitionToOrderedCount}
import org.apache.spark.rdd.RDD

class DisCGS( dataRDD: RDD[(Int, DenseVector[Double])],
              prior: NormalInverseWishart,
              n: Int,
              masterAlphaPrior: Double,
              workerAlphaPrior: Double,
              computeLikelihood: Boolean = false) extends Serializable {

  var partitionEveryIteration:List[List[Int]] = List(List.fill(n)(0))
  var likelihoodEveryIteration: List[Double] = List()

  // Initialize dpm in each worker
  val workersDPM = dataRDD.mapPartitionsWithIndex { (index, Data) => {
    val dataItList = convertIteratorToList(Data)
    Iterator(new WorkerCGS(workerID = index, indices = dataItList._1, data = dataItList._2, alpha = workerAlphaPrior, prior = prior))
  }
  }.persist

  def run(nIter:Int) {
  //Run dpm in each worker
  var workerResults = workersDPM.map(e => List(e.run(1))).persist.reduce((x, y) => x ++ y).sortBy(_._1)
  var processedResults: List[(Int, Int, (DenseVector[Double], DenseMatrix[Double]), Int)] = List()
  for (r <- workerResults) {
    val workerID = r._1
    val sufficientStatistics = r._3
    val countCluster = r._4
    for (k <- countCluster.indices) {
      processedResults = processedResults :+ (workerID, k, sufficientStatistics(k), countCluster(k))
    }
  }
  var it = 1
  while (it < nIter) {

    //Initialize DPM at master level
    val masterDPMM = new MasterCGS(workerResults = processedResults, alpha = masterAlphaPrior, prior = prior)

    // Run DPM at master level
    val masterResults = masterDPMM.run(1)
    val clusters: List[Int] = masterResults._1
    val NIWParams = masterResults._2

    val allResults = processedResults.indices.map(i => (processedResults(i)._1, processedResults(i)._2, NIWParams(clusters(i)), clusters(i))).toList

    workerResults = workersDPM.map(dpm => {
      dpm.updateDPMWithMasterResults(allResults)
      List(dpm.run(1))
    }).reduce((x, y) => x ++ y).sortBy(_._1)

    processedResults = List()

    for (r <- workerResults) {
      val workerID = r._1
      val sufficientStatistics = r._3
      val countCluster = r._4
      for (k <- countCluster.indices) {
        processedResults = processedResults :+ (workerID, k, sufficientStatistics(k), countCluster(k))
      }
    }
    val glob = workersDPM.map(dpm => dpm.getGlobalResults()).reduce((x, y) => x ++ y).sortBy(_._1)
    partitionEveryIteration = partitionEveryIteration :+ glob.map(_._2)

    if(computeLikelihood){
    val globalCounCluster: List[Int] = partitionToOrderedCount(partitionEveryIteration.last)
    val components = NIWParams.map(_.sample())
      val data: List[DenseVector[Double]] = dataRDD.collect().toList.map(_._2)
      val likelihood = prior.likelihood(alpha = masterAlphaPrior,
        data = data,
        membership = partitionEveryIteration.last,
        countCluster = globalCounCluster,
        components = components)
      likelihoodEveryIteration = likelihoodEveryIteration :+ likelihood
    }
    it = it + 1
  }
  }
}
