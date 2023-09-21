import GS.{DisCGS, NormalInverseWishart}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import smile.validation.{NormalizedMutualInformation, adjustedRandIndex}
import org.json4s.DefaultFormats
import org.json4s.jackson.Serialization.writePretty

import java.io.{File, PrintWriter}
import Common.Tools.getDatasetInformation
import org.json4s.native.JsonMethods
import breeze.linalg.{DenseVector, linspace}

import scala.io.Source

object Main {
  def main(args: Array[String]): Unit = {
    implicit val formats = DefaultFormats

     // list of possible datasets:
     // "EngyTime_4096_2_2" "mnist_70000_8_10" "fashion-mnist_70000_8_10" "balanced_131600_8_47" "digits_280000_8_10"
     // "urbanGB_360177_2_469" "emnist_letter_103600_8_26" "c_synthetic_10000_2_6" "synthetic_20000_2_10"
     // "synthetic_40000_2_10" "synthetic_60000_2_10" "synthetic_80000_2_10" "synthetic_100000_2_10" "synthetic_1000000_2_10"


    val nameDataset: String = args(0)
    val nbWorkers: Int = args(1).toInt
    val nRuns: Int = args(2).toInt
    val computeLikelihood: Boolean = args(3).toBoolean


    val datasetsHandelInfo: Map[String, Map[String, Any]] = JsonMethods.parse(Source.fromFile("../datasets/experiences_config.json").reader()).extract[Map[String, Map[String, Any]]]
    val conf = new SparkConf().setMaster(s"local[$nbWorkers]").setAppName("Distributed Inference for DPMM")
    val sc = new SparkContext(conf)



     var partitionsList: List[List[Int]] = List()
     var likelihoodsList: List[List[Double]] = List()

     var runTimesList: List[Double] = List()
     var ariList: List[Double] = List()
     var nmiList: List[Double] = List()
     var infClustList: List[Int] = List()

      // get dataset information
     val datasetInfo = getDatasetInformation(datasetName = nameDataset, datasetsHandelInfo = datasetsHandelInfo).get
     val datasetType: String = datasetInfo("name_dataset").toString
     val nObservations: Int = datasetInfo("number_of_observations").toString.toInt
     val dimension: Int = datasetInfo("dimension").toString.toInt
     val nClusters: Int = datasetInfo("number_of_clusters").toString.toInt
    val workerAlphaPrior: Double = datasetInfo("alpha_worker").toString.toDouble
    val masterAlphaPrior: Double = datasetInfo("alpha_master").toString.toDouble

    Logger.getLogger("org").setLevel(Level.ERROR)


      val dataJ = JsonMethods.parse(Source.fromFile("../datasets/data" ++ s"/${nameDataset}.json").reader()).children

      val data: List[DenseVector[Double]] = dataJ(0).children.map(y => DenseVector(y.children.map(z => z.extract[Double]): _*))
      val trueLabels: List[Int] = dataJ(1).children.map(z => z.extract[Int])


      val dataRDD = sc.parallelize(data.zipWithIndex.map(e => (e._2, e._1)), nbWorkers)

      val empiricMean = Common.Tools.mean(data)
      val empiricCovariance = Common.Tools.covariance(data, empiricMean)
      val prior = new NormalInverseWishart(empiricMean, 1D, empiricCovariance, data.head.length + 1)

      for (run <- 0 until nRuns) {


        val t0 = System.nanoTime()
        val DisDPMM = new DisCGS(masterAlphaPrior = masterAlphaPrior, workerAlphaPrior = workerAlphaPrior, prior = prior, dataRDD = dataRDD, n = nObservations, computeLikelihood = computeLikelihood)
        DisDPMM.run(nIter = 100)
        val t1 = System.nanoTime()

        val partitionEveryIteration = DisDPMM.partitionEveryIteration
        likelihoodsList = likelihoodsList :+ DisDPMM.likelihoodEveryIteration

        partitionsList = partitionsList :+ partitionEveryIteration.last
        runTimesList = runTimesList :+ (t1 - t0) / 1e9D
        ariList = ariList :+ adjustedRandIndex(partitionEveryIteration.last.toArray, trueLabels.toArray)
        nmiList = nmiList :+ NormalizedMutualInformation.sum(partitionEveryIteration.last.toArray, trueLabels.toArray)
        infClustList = infClustList :+ partitionEveryIteration.last.max

        println(s">>>>>> Launch: $run")
        println(s"Runtime: ${runTimesList.last}")
        println(s"ARI: ${ariList.last}")
        println(s"NMI: ${nmiList.last}")
        println(s"Number of inferred clusters: ${infClustList.last}")
        if(computeLikelihood){
        println(s"Log-likelihood: ${likelihoodsList.last.last}")
        }
      }

      val results = Map(
        "dataset name" -> datasetType,
        "nObservations" -> nObservations,
        "nClusters" -> nClusters,
        "dimension" -> dimension,
        "NbWorkers" -> nbWorkers,
        "runtimeList" -> runTimesList,
        "infClust" -> infClustList,
        "ariList" -> ariList,
        "nmiList" -> nmiList,
        "workerAlphaPrior" -> workerAlphaPrior,
        "masterAlphaPrior" -> masterAlphaPrior,
        "partitionsList" -> partitionsList,
        "likelihoodList" -> likelihoodsList
      )



    val finalResults = writePretty(results)
    val f = new File(s"../results/Experience_results_${nameDataset}.json")
    val w = new PrintWriter(f)
    w.write(finalResults)
    w.close()
  }
}