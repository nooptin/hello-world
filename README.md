#Abstract—Internet-based services monitor and detect anomalies
on KPIs (Key Performance Indicators, say CPU utilization,
number of queries per second, response latency) of their applications
and systems in order to keep their services reliable. This
paper identifies a common, important, yet little-studied problem
of KPI anomaly detection: rapid deployment of anomaly detection
models for large number of emerging KPI streams, without manual
algorithm selection, parameter tuning, or new anomaly labeling for
any newly emerging KPI streams. We propose the first framework
ADS (Anomaly Detection through Self-training) that tackles the
above problem, via clustering and semi-supervised learning. Our
extensive experiments using real-world data show that, with the
labels of only the 5 cluster centroids of 70 historical KPI streams,
ADS achieves an averaged best F-score of 0.92 on 81 new KPI
streams, almost the same as a state-of-art supervised approach,
and greatly outperforming a state-of-art unsupervised approach
by 61.40% on average.

摘要-基于互联网服务监视和检测异常关于系统及应用的KPIs（关键性能指标，比如CPU利用率，每秒查询数,（响应延迟）以确保其服务可靠。该论文确定一个常见的，重要的，但很少研究的KPI异常检测问题：异常检测的快速部署大量新兴KPI流的模型，无需选择手动算法、参数调整，或任何新出现的KPI流的异常标记。我们提出第一个框架
ADS（通过自我训练进行异常检测）解决以上问题，并通过聚类和半监督学习。我们使用真实世界数据做了实大量验表明：仅标记70个历史KPI流的5个群集质心，在81个新的关键绩效指标中，自我训练异常检测的平均最佳F-score为0.92，几乎和最先进的监督方法一样，比最先进的无监督方法有更好的表现，平均增长61.40%。
I. INTRODUCTION 
一、导言
Internet-based services (e.g., online games, online shopping,
social networks, search engine) monitor KPIs (Key Performance
Indicators, say CPU utilization, number of queries per
second, response latency) of their applications and systems in
order to keep their services reliable. Anomalies on KPI (e.g.,
a spike or dip in a KPI stream) likely indicate underlying
failures on Internet services [1]–[5], such as server failures,
network overload, external attacks, and should be accurately
and rapidly detected.
基于互联网的服务（例如，在线游戏、在线购物，社交网络、搜索引擎）监控关键绩效指标指标，比如CPU利用率，每个第二，响应延迟）
以保证他们的服务可靠。关键绩效指标异常（例如。，KPI流中的峰值或下降）可能表示潜在，的Internet服务故障[1]–[5]，例如服务器故障，网络过载，外部攻击，并且应该准确 很快就被发现了。

Despite the rich body of literature in KPI anomaly detection[2], [6]–[12], there remains one common and important scenario that has not been studied or well-handled by any of these approaches. Specifically, when large number of KPI streams emerge continuously and frequently, operators need to deploy accurate anomaly detection models for these new KPI streams as quickly as possible (e.g., within 3 weeks at most),in order to avoid that Internet-based services suffer from false alarms (due to low precision) and/or missed alarms (because of low recall) and in turn impact on user experience and revenue. Large number of new KPI streams emerge due to the following two reasons. First, new products can be frequently launched, such as in gaming platform. For example, in a top gaming company G studied in this paper, on average over ten new games are launched per quarter, which results in more than 6000 new KPI streams per 10 days on average
尽管在KPI异常检测[2]、[6]–[12]中有大量的文献，但仍然有一个常见而重要的场景没有被这些方法研究或很好地处理。具体来说，当大量的KPI流不断频繁出现时，运营商需要尽快为这些新的KPI流部署准确的异常检测模型（例如，最多3周内），以避免基于Internet的服务出现错误报警（由于精度低）和/或漏报（因为低召回率），进而影响用户体验和收入。由于以下两个原因，出现了大量新的KPI流。首先，新产品可以频繁推出，比如在游戏平台上。例如，在本文所研究的一家顶级游戏公司G中，平均每个季度有超过10款新游戏推出，平均每10天就有超过6000款新的KPI流

Second, with the popularity of DevOps and micro-service, software upgrades become more and more frequent [13], many of which result in the pattern changes of existing KPI streams, making the previous anomaly detection algorithms/parameters outdated.
其次，随着DevOps和微服务的普及，软件升级变得越来越频繁（13），从而导致现有KPI流的模式改变，使得以前的异常检测算法/参数过时。

Unfortunately, none of the existing anomaly detection approaches,
including traditional statistical algorithms, supervised learning, and unsupervised learning, are feasible to deal with the above scenario well. For traditional statistical algorithms [6]–[9], to achieve the best accuracy, operators have to manually select an anomaly detection algorithm and tune its parameters for each KPI stream, which is infeasible for the large number of emerging KPI streams. Supervised learning based methods [2], [10] require manually labeling anomalies for each new KPI stream, which is not feasible for the large number of emerging KPI streams either. Unsupervised
learning based methods [11], [12] do not require algorithm selection, parameter tuning, or manual labels, but they either
suffer from low accuracy [14] or require large amounts of training data for each new KPI stream (e.g., six months worth of data) [12], which do not satisfy the requirement of rapid deployment (e.g., within 3 weeks) of accurate anomaly detection. 
不幸的是，没有现有的异常检测方法，

包括传统的统计算法、有监督学习和无监督学习，都可以很好地解决上述问题。对于传统的统计算法[6]-[9]，为了达到最佳的精度，操作员必须手动选择一个异常检测算法，并为每个KPI流调整其参数，这对于大量新兴的KPI流是不可行的。基于监督学习的方法[2]，[10]要求为每个新的KPI流手动标记异常，这对于大量出现的KPI流也是不可行的。无监督 基于学习的方法[11]，[12]不需要算法选择、参数调整或手动标签，但它们也可以 存在精度低[14]的问题，或者需要为每个新的关键绩效指标流提供大量的培训数据（例如，价值六个月的数据）[12]，这些数据无法满足快速部署（例如，在3周内）准确异常检测的要求。

In this paper, we propose ADS, the first framework that enables the rapid deployment of anomaly detection models (say at most 3 weeks) for large number of emerging KPI streams, without manual algorithm selection, parameter tuning, or new anomaly labeling for any newly emerging KPI streams.
在本文中，我们提出了自我训练异常检测，这是第一个框架，它可以快速部署大量新出现的KPI流的异常检测模型（最多3周），而无需对任何新出现的KPI流进行手动算法选择、参数调整或新的异常标记。

Our idea of ADS is based on the following two observations. (1) In practice, many KPI streams (e.g., the number of queries per server in a well load balanced server cluster) are similar due to their implicit associations and similarities, thus potentially we can use the similar anomaly detection algorithms and parameters for these similar KPI streams. (2) Clustering methods such as ROCKA [15] can be used to cluster many KPI streams into clusters according to their similarities. The number of clusters are largely determined by the nature of the service (e.g., shopping, gaming, social network, search) and the type of KPIs (e.g., number of queries, CPU usage,
memory usage), but not by the scale of the entire system. Thus for a given service, the number of clusters can be orders of magnitude smaller than the number of KPI streams, and there is a good chance that a newly emerging KPI stream falls into one of the existing clusters resulted from historical KPI streams.
我们的自我训练异常检测理念基于以下两个观察。（1）在实际应用中，许多关键绩效指标流（如负载均衡的服务器集群中每台服务器的查询数）由于其隐含的关联性和相似性而相似，因此我们有可能对这些相似的关键绩效指标流使用相似的异常检测算法和参数。（2）可以使用ROCKA[15]等聚类方法，根据多个KPI流的相似性将其聚类到多个集群中。集群的数量在很大程度上取决于服务的性质（例如购物、游戏、社交网络、搜索）和kpi的类型（例如查询的数量、CPU的使用量，

内存使用率），但不是整个系统的规模。因此，对于给定的服务，簇的数量可以比KPI流的数量小几个数量级，并且很有可能新出现的KPI流落入由历史KPI流引起的现有集群中的一个。

Utilizing the above two observations, ADS proposes to (1) cluster all existing/historical KPI streams into clusters, (2) manually label the anomalies of all cluster centroids, (3) assign each newly emerging KPI stream into one of the existing clusters, and (4) combine the data of the new KPI stream (unlabeled) and it’s cluster centroid (labeled) and use semisupervised learning [16] to train a new model for each new KPI stream. During semi-supervised learning, ADS’s base model is supervised learning such as [2], thus is able to avoid algorithm selection and parameter tuning. Semi-supervised learning can train a new model for a new KPI stream using an existing labeled KPI stream as long as these two KPI streams have similar data distribution, which is the case since they are in the same cluster. This way, ADS enjoys the
benefits of supervised learning, yet only needs to label a
much smaller number of historical KPI streams (i.e., cluster
centroids) without labeling new KPI streams. Unsupervised
learning based methods are not selected as the base model
of ADS due to low accuracy [14] or the requirement of long
period of data for each new KPI stream (e.g., six months worth
of data) [12], but they are nonetheless compared with ADS in
Section IV.
利用上述两个观察，自我训练异常检测提出（1）将所有现有/历史KPI流聚类成集群，（2）手动标记所有聚类中心的异常，（3）将每个新出现的KPI流分配到现有的聚类中的一个，以及（4）结合新的KPI流（未标记）的数据和它的聚类质心（标记）。并使用半监督学习[16]为每个新的KPI流训练一个新模型。在半监督学习过程中，自我训练异常检测的基础模型是监督学习，如[2]，这样可以避免算法选择和参数调整。半监督学习可以使用现有的标记KPI流来为新的KPI流建立新的模型，只要这两个KPI流具有类似的数据分布，这是因为它们在同一个集群中。这样，自我训练异常检测就可以享受监督学习的好处，但只需要标注
历史KPI流（即集群）的数量要少得多质心）没有标记新的KPI流。无监督未选择基于学习的方法作为基础模型 由于精度低[14]或长
每个新的KPI流的数据周期（例如，值6个月数据的[12]，但它们与第四节。

The contributions of this paper are summarized as follows:
本文的贡献总结如下：

_ To the best of our knowledge, this paper is the first to identify the common and important problem of rapid deployment of anomaly detection models for large number of emerging KPI streams, without manual algorithm selection, parameter tuning, or new anomaly labeling for any newly generated KPI streams, and proposes the first framework ADS that tackles this problem.
据我们所知，本文首先确定了大量新出现的KPI流快速部署异常检测模型的常见而重要的问题，不需要对任何新生成的KPI流进行手动算法选择、参数调整或新的异常标记，并提出了解决这一问题的第一个框架自我训练异常检测。

_ To the best of our knowledge, this paper is the first to apply semi-supervised learning to the KPI anomaly detection problem. We adopt a robust semi-supervised learning model, contrastive pessimistic likelihood estimation (CPLE), which is suitable for KPI anomaly detection and only requires similar (not necessarily the same) data distribution between the existing labeled KPI stream and the new KPI stream.
据我们所知，本文首次将半监督学习应用于KPI异常检测问题。我们采用一种鲁棒的半监督学习模型，对比悲观似然估计（CPLE），它适合于KPI异常检测，并且只需要在现有标记的KPI流和新的KPI流之间相似（不一定相同）的数据分布。

_ We conduct extensive experiments using 70 historical KPI streams and 81 new KPI streams from a top global online game service G. With the labels of only the 5 cluster centroids of 70 historical KPI streams, ADS achieves an averaged best F-score of 0.92 on 81 new KPI streams, almost the same as the state-of-art supervised approach [2] which requires the labels for all 81 new KPI streams, and greatly outperforms an unsupervised approach Isolation Forest [14] by 360% and the state-of-art unsupervised
approach Donut [12] by 61.40% on average.

我们使用来自全球顶级在线游戏服务G的70个历史KPI流和81个新KPI流进行了广泛的实验。仅使用70个历史KPI流的5个簇中心的标签，自我训练异常检测在81个新KPI流上获得了0.92的平均最佳F分数，几乎与最新监管的方法[2]相同，后者要求为所有81个新的KPI流添加标签，并且大大优于无监管的方法隔离林[14]360%和最新无监管的方法

接近甜甜圈[12]平均61.40%。

The rest of this paper is organized as follows. In Section II, we review the background, related works and motivation. The 
framework of ADS is introduced in Section III. We report the
experimental results of evaluating ADS in Section IV. Finally,
we give a conclusion of this paper in Section V.
本文的其余部分安排如下。
在第二节中，我们回顾了研究背景、相关工作和动机。第三节介绍了自我训练异常检测的框架，第四节给出了评估自我训练异常检测的实验结果，最后在第五节，我们给出了本文的结论。
 
Fig. 1: Examples of anomalies in KPI streams. The red parts in the KPI stream denote anomalous points, and the orange part denotes missing points (filled with zeros).
图1:KPI流中异常的示例。KPI流中的红色部分表示异常点，橙色部分表示缺失点（用零填充）。

II. BACKGROUND背景
A. Anomaly Detection for KPI Streams 
A．KPI流的异常检查

A KPI stream of an Internet-based service is a time series with the format of (timestamp, value). It is essentially monitoring data collected from Simple Network Management Protocol (SNMP), syslogs, web access logs or other data sources [17]–[19]. It can be denoted as xt􀀀m+1; : : : ; xt, where xi is a monitoring value at time i; i 2 [t 􀀀 m + 1; t], t is the present time, and m is the length of the KPI stream.
基于Internet的服务的KPI流是格式为（时间戳、值）的时间序列。它实际上是监视从简单网络管理协议（SNMP）、系统日志、web访问日志或其他数据源收集的数据[17]–[19]。它可以被表示为：+ + 1：：：：；；XT，其中XI是i时的一个监视值；i 2 T + 1；t；t是当前时间，m是KPI流的长度。

Anomalous data points of a KPI stream usually have different
data characteristics from those of normal data points. For example, a spike, a level shift or a dip in a KPI stream likely indicates an anomaly. Figure 1 shows three examples of anomalies in KPI streams. Anomaly detection for the KPI stream xt􀀀m+1; : : : ; xt is to determine whether xt is an anomalous data point (let yt = 1 denote an anomalous data point and yt = 0 denote a normal one).
KPI流的异常数据点通常具有不同的

正常数据点的数据特征。例如，KPI流中的峰值、级别偏移或下探可能表示异常。图1显示了KPI流中异常的三个示例。KPI流xtm+1的异常检测；：：；xt用于确定xt是否为异常数据点（让yt=1表示异常数据点，yt=0表示正常数据点）。

Most anomaly detection algorithms, including traditional statistical algorithms, supervised learning based methods and unsupervised learning based methods, compute an anomaly score for a data point to denote how likely this data point is anomalous. Operators then set a threshold to determine whether each data point is anomalous or not. That is, only if the anomaly score at time t exceeds this threshold, xt will be regarded as an anomalous data point.
大多数异常检测算法，包括传统的统计算法、基于监督学习的方法和基于无监督学习的方法，计算数据点的异常分数，以表示该数据点异常的可能性。然后，操作员设置一个阈值来确定每个数据点是否异常。也就是说，只有在时刻t的异常得分超过该阈值时，xt才被视为异常数据点。

From the above definition, we can see that anomaly detection
for a KPI stream is essentially a two-class classification problem – classifying a data point into an anomalous data point or a normal one. Consequently, we can use the intuitive classification metrics of two-class classification methods, including precision, recall, and F-score, to evaluate the performance of anomaly detection algorithms.
从上面的定义，我们可知对于KPI流异常检测来说，本质上是一个两类分类问题-将数据点分类为异常数据点或正常数据点。因此，我们可以使用两种分类方法的直观分类度量，包括精确性、召回率和F-score，来评估异常检测算法的性能。
B. Anomaly Detection Methods for KPI Streams
B．KPI流异常检测方法

Anomaly detection for KPI streams deals with the task of recognizing unexpected data points from normal behavior. Over the years, diverse traditional statistical algorithms have been applied for KPI anomaly detection, including SVD [6], Wavelet [7], ARIMA [8], Time Series Decomposition [1], Holt-Winters [9], etc. Each of the above algorithms computes an anomaly score for each data point in a KPI stream on the basis of simple statistical assumptions. To achieve the best accuracy, operators have to manually select an anomaly detection algorithm and tune its parameters for each KPI stream. Since it is often the case that a large number of newly emerging KPI streams have to be carefully monitored [13],manual algorithm selection and parameter tuning for every newly emerging KPI stream becomes infeasible.
KPI流异常检测处理从正常行为中识别意外数据点的任务。多年来，各种传统的统计算法被应用于KPI异常检测，包括SVD[6]、小波[7]、ARIMA[8]、时间序列分解[1]、Holt-Winters[9]等，以上算法都是在简单的统计假设的基础上计算KPI流中每个数据点的异常得分。为了获得最佳的准确性，操作员必须手动选择异常检测算法，并为每个KPI流调整其参数。由于经常需要仔细监视大量新出现的KPI流[13]，因此为每个新出现的KPI流进行手动算法选择和参数调整变得不可行。
To address the problem posed by algorithm selection and 
parameter tuning, several supervised learning based methods such as EGADS [10] and Opprentice [2] have been proposed. For each KPI stream, these methods learn (traditional statistical) algorithm selection and parameter tuning from operators’ manual labels of KPI anomalies. Obviously, manual anomaly labeling for a large number of newly emerging KPI streams is not feasible either.
解决算法选择和 提出了参数整定、基于监督学习的方法，如EGADS[10]和oppertice[2]。对于每个KPI流，这些方法从操作员的KPI异常手动标签中学习（传统的统计）算法选择和参数调整。显然，对大量新出现的KPI流进行手动异常标记也是不可行的。

Unsupervised learning has emerged as a promising field in KPI anomaly detection. For example, isolation based methods [11] and variational autoencoders (VAE) [12] are applied in detecting anomalies in (KPI) streams. These methods are trained without manual labels, and thus they can be applied for large volume of KPI streams. However, isolation based methods suffer from low accuracy [14] (see Section IV-C for more details). In addition, Donut [12], which is based on VAE, requires a long period (say six months) of training data for newly emerging KPI streams. During this period, the Internet-based services may suffer from false alarms (due to low precision) and/or missed alarms (because of low recall) and in turn impact user experience and revenue.
无监督学习已成为KPI异常检测中一个很有前途的研究领域。例如，基于隔离的方法[11]和变分自动编码器（VAE）[12]被应用于检测（KPI）流中的异常。这些方法不需要人工标注，因此可以应用于大量的KPI流。然而，基于隔离的方法精度较低[14]（详见第IV-C节）。此外，基于VAE的Donut[12]需要一段很长的时间（比如说六个月）来为新出现的KPI流提供培训数据。在此期间，基于互联网的服务可能会出现错误警报（由于精度低）和/或错过警报（由于召回率低），进而影响用户体验和收入。

As discussed above, existing anomaly detection algorithms for KPI streams suffer from manual algorithm selection and parameter tuning (traditional statistical algorithms), or manual anomaly labeling (supervised learning based methods), or low accuracy in real-world Internet-based service (unsupervised learning based methods like isolation forests [11]), or long period of training time for newly generated KPI streams (unsupervised learning based methods like Donut [12]).
如上所述，现有的KPI流异常检测算法遭受人工算法选择和参数调整（传统统计算法），或手动异常标记（基于监督学习的方法），或基于真实世界的基于互联网的服务的低准确度（基于隔离森林的无监督学习方法）。[11]），或新生成的KPI流的长时间培训（无监督的基于学习的方法，如甜甜圈[12]）。

Semi-supervised learning [16] is halfway between supervised and unsupervised learning. It uses unlabeled data to modify either parameters or models obtained from labeled data alone to maximize the learning performance. [20]–[22] use semi-supervised learning for anomaly detection in other domains, but are not designed for KPI streams (time series).
半监督学习[16]介于监督学习和非监督学习之间。它使用未标记的数据来修改从标签数据单独获得的参数或模型，以最大限度地提高学习性能。[20]–[22]在其他领域使用半监督学习进行异常检测，但不是为KPI流（时间序列）设计的。

C. Clustering of KPI Streams
C．KPI流的聚类

Millions of KPI streams bring a huge challenge to KPI anomaly detection. Luckily, many KPI streams are similar because of their implicit associations and similarities. For example, Figure 2 shows two KPI streams of response latency collected from two different applications, and these two KPI streams are quite similar in shape. If we can identify similar KPI streams, and group the huge volume of KPI streams into a few clusters, we can reduce the overhead in anomaly detection. 
数以百万计的KPI流给KPI异常检测带来了巨大的挑战。幸运的是，许多KPI流由于其隐式关联和相似性而相似。例如，图2显示了从两个不同的应用程序收集的两个响应延迟的KPI流，这两个KPI流的形状非常相似。如果我们能够识别出相似的KPI流，并将大量的KPI流分成几个簇，就可以减少异常检测的开销。

Time series clustering is a popular field which has caught lots of attention in the past 20 years. [23] summarized a large number of methods on this topic, most of which are designed for smooth and idealized data. However, the large number of spikes, dips and level shifts in KPI streams can significantly change the shape of KPI streams. Therefore, the above methods do not perform good for KPI streams.

时间序列聚类是近20年来备受关注的一个研究领域。[23]总结了大量关于这一主题的方法，其中大多数是为平滑和理想化的数据而设计的。然而，KPI流中的大量尖峰、尖峰和水平偏移会显著改变KPI流的形状。因此，上述方法对KPI流不起作用
 
Fig. 2: The response latency streams of two different applications.
图2：两个不同应用的响应延迟流。
In this work, we adopt ROCKA [15], a rapid clustering algorithm for KPI streams based on their shapes. It applies moving average to extract baselines which successfully reduce the biases of noises and anomalies. In addition, it uses shapebased distance as the distance measure, and reduces its algorithm complexity to O(mlog(m)) using Fast Fourier Transform. Finally, it uses DBSCAN to cluster KPI streams and chooses the centroid for every cluster. Extensive experiments in [15] have demonstrated ROCKA’s superior performance in clustering KPI streams for large Internet-based services. Please note that applying ROCKA to cluster KPI streams is not our contribution.
在本文中，我们采用了ROCKA[15]，一种基于KPI流形状的快速聚类算法。采用移动平均法提取基线，有效地降低了噪声和异常的偏差。此外，它使用基于形状的距离作为距离度量，并使用快速傅立叶变换将其算法复杂度降低到O（MLO（m））。最后，使用DBSCAN对KPI流进行聚类，并为每个聚类选择质心。在[15]中的大量实验已经证明了ROCKA在为大型基于互联网的服务聚类KPI流方面的优越性能。请注意，将ROCKA应用于集群KPI流不是我们的贡献。

III. FRAMEWORK OF ADS
III. 自我训练异常检测框架

As aforementioned, we propose ADS, a semi-supervised learning based anomaly detection framework for KPI streams, to tackle the problem of rapid deployment of anomaly detection models for large number of emerging KPI streams, without manual algorithm selection, parameter tuning, or new anomaly labeling for any newly generated KPI streams.
如前所述，我们提出了一个半监督学习的KPI流异常检测框架自我训练异常检测，以解决对大量新出现的KPI流快速部署异常检测模型的问题，而无需对任何新生成的KPI流进行手动算法选择、参数调整或新的异常标记。

Figure 3 shows the framework of ADS. For historical KPI streams, ADS preprocesses them with filling missing points and standardization, clusters the preprocessed KPI streams using ROCKA (Section III-A), and extracts features (namely,output results of different anomaly detection algorithms and parameters) for the KPI streams on cluster centroids (Section III-B). Similarly, when a new KPI stream is generated, ADS preprocesses it, and classifies it into one of the above clusters (Section III-A), after which the features of this new KPI stream are extracted (Section III-B). Based on the features of the new KPI stream, and the features and labels of the new KPI stream’s cluster centroid, ADS trains a semi-supervised learning model using the CPLE algorithm (Section III-C). Finally, ADS detects anomalies in the new KPI stream based on the above model and the severity threshold (aThld henceforth).

图3显示了自我训练异常检测的框架。对于历史的关键绩效指标流，自我训练异常检测使用填充缺失点和标准化对其进行预处理，使用ROCKA（第III-A节）对预处理的关键绩效指标流进行聚类，并提取特征（即，集群质心上KPI流的不同异常检测算法和参数的输出结果（第III-B节）。类似地，当生成一个新的KPI流时，自我训练异常检测会对其进行预处理，并将其分类为上述集群之一（第III-a节），然后提取该新KPI流的特征（第III-B节）。根据新KPI流的特点，以及新KPI流聚类中心的特征和标记，自我训练异常检测采用CPLE算法训练半监督学习模型（第三节C）。最后，自我训练异常检测基于上述模型和严重性阈值（aThld）检测新KPI流中的异常。

A. Preprocessing and Clustering
A．预处理和聚类
In Internet-based services, monitoring system malfunctions and/or operator misconfigurations, though occur infrequently, can lead to missing points in KPI streams. These missing points can bring significant biases to ADS. Therefore, we fill these missing points using linear interpolation following [15].
在基于Internet的服务中，监视系统故障和/或操作员错误配置（虽然很少发生）可能会导致缺少关键绩效指标流中的点。这些缺失点会给自我训练异常检测带来显著的偏差，因此，我们在[15]之后使用线性插值来填充这些缺失点。

 
Fig. 3: The framework of ADS
图3: 自我训练异常检测的框架
In addition, to make KPI steams of different amplitudes and/or length scales comparable, we also standardize KPI streams. This way, different KPI streams are clustered based on their shapes,    rather than the absolute values of amplitudes and/or length scales.
此外，为了使不同幅度和/或长度尺度的KPI流具有可比性，我们还对KPI流进行了标准化。这样，不同的KPI流将基于其形状而不是振幅和/或长度刻度的绝对值进行集群。
As discussed in section II-C, ADS adopts ROCKA [15] to group KPI streams into a few clusters, and obtains a centroid KPI stream for each cluster.
如第II-C节所述，自我训练异常检测采用ROCKA[15]将KPI流分组为几个集群，并为每个集群获得一个质心KPI流。

B. Feature Extraction
B.特征提取
When training based on historical data, since the KPI stream on the centroid of each cluster represents the cluster’s characteristics, we extract the features of the KPI stream on each cluster centroid. In addition, when training based on new data, we extract the features of the newly generated KPI stream. The features of the new KPI stream, the features and the anomaly labels of the new KPI stream’s cluster centroid, together form the training set.
在基于历史数据的训练中，由于每个聚类中心上的KPI流代表了聚类的特征，因此我们提取每个聚类中心上KPI流的特征。另外，在基于新数据进行训练时，我们提取新生成的KPI流的特征。新KPI流的特征、新KPI流的簇质心的特征和异常标签共同构成训练集。

Here we introduce how to extract features for KPI streams. As is with [2], we use the output results of different anomaly detectors (namely, the anomaly severities measured by anomaly detectors) as the features of KPI streams. This way, each detector serves as a feature extractor.
这里我们介绍如何提取KPI流的特征。与文献[2]一样，我们使用不同异常检测器的输出结果（即异常检测器测量的异常严重度）作为KPI流的特征。这样，每个检测器都充当一个特征提取器。

When an anomaly detector receives an incoming data point of a KPI stream, it internally produces a non-negative value, called severity, to measure how anomalous that data point is. For example, historical average [24] applies how many times of standard deviations the point is away from the mean as the severity, based on the assumption that the KPI stream data follows Gaussian distribution; Holt-Winters [9] uses a residual error (namely, the absolute difference between the actual value and the forecast value of each data point) to measure the severity. In addition, most anomaly detectors are parameterized and have a set of internal parameters (say, historical average has one parameter of window length, and Holt-Winters has three parameters f_; _; g). As a result, both detectors and their internal parameters decide the severity of a given data point.
当异常检测器接收到KPI流的传入数据点时，它会在内部生成一个称为severity的非负值，以测量该数据点的异常程度。例如，基于假设KPI流数据遵循高斯分布，历史平均值[24]将该点偏离平均值的多少倍作为严重性；Holt Winters[9]使用残差（即，每个数据点的实际值和预测值之间的绝对差），以测量严重性。此外，大多数异常检测器都是参数化的，并且有一组内部参数（例如，历史平均值有一个窗口长度参数，而Holt-Winters有三个参数fúug）。因此，探测器及其内部参数决定了给定数据点的严重性。

In this work, for each (parameterized) anomaly detector, ADS samples its parameters to generate one or more “fixed” anomaly detectors. This way, an anomaly detector with specific sampled parameters acts as a feature extractor as follows:
在这项工作中，对于每个（参数化）异常检测器，自我训练异常检测对其参数进行采样，以生成一个或多个“固定”异常检测器。这样，具有特定采样参数的异常检测器充当如下特征提取器：
 
The feature extraction, training, and classification (detection) in ADS are all designed to work with individual data points, not anomaly windows, so that the semi-supervised learning algorithm can have enough data for training. Another benefit of this design choice is that the classifier can detect anomalies fast on each data point.
自我训练异常检测中的特征提取、训练和分类（检测）都是针对单个数据点而非异常窗口设计的，这样半监督学习算法就可以有足够的数据进行训练。这种设计选择的另一个好处是，分类器可以在每个数据点上快速检测异常。

TABLE I: Detectors and sampled parameters used in ADS. Some abbreviations are MA (moving average), EWMA (exponentially weighted MA), TSD (time series decomposition), SVD (singular value decomposition), win(dow), and freq(uency).
表一：自我训练异常检测中使用的检测器和采样参数。一些缩写是MA（移动平均）、EWMA（指数加权MA）、TSD（时间序列分解）、SVD（奇异值分解）、win（dow）和freq（uency）。
 
Following [2], we implement 14 widely used anomaly detectors in ADS. All the 14 anomaly detectors and their sample parameters are shown in Table I.
[2]之后，我们在自我训练异常检测中实现了14个广泛使用的异常检测器，所有14个异常检测器及其样本参数如表1所示。

C. Semi-Supervised Learning
C.半监督学习
After extracting features of anomaly detectors for both the labeled KPI streams on cluster centroids and the unlabeled new KPI stream, we try to learn a model which is based on both labeled and unlabeled data, namely a semi-supervised learning model.
在对聚类中心上标记的KPI流和未标记的新KPI流提取异常检测器特征后，我们尝试学习一个基于标记和未标记数据的模型，即半监督学习模型。

As is summarized in [16], different semi-supervised learning models have different advantages and disadvantages. Among these methods, self-training based methods [20] apply an existing model to “label” unlabeled data, and employ the newly labeled data together with the actual labeled data to retrain the model until the prediction result no longer changes or iteration ends.
正如[16]所总结的，不同的半监督学习模型有不同的优缺点。在这些方法中，基于自训练的方法〔20〕应用现有模型来“标记”未标记数据，并将新标记的数据与实际标记的数据一起重新训练模型，直到预测结果不再改变或迭代结束。

In this work, we adopt CPLE [28], an extension model of self-training. CPLE is a resilient semi-supervised learning framework for any supervised learning classifier (base-model henceforth) including random forest, SVM, decision tree, etc. It takes the prediction probabilities of base-model as input to fully utilize the unlabeled data. CPLE has the three following advantages:
在这项工作中，我们采用了CPLE[28]，一个自我训练的扩展模型。CPLE是一个弹性半监督学习框架，适用于任何监督学习分类器（包括随机林、支持向量机、决策树等），它以基础模型的预测概率为输入，充分利用未标记数据。CPLE具有以下三个优点：
•	CPLE is flexible to change base-model, so we can set its base-model to achieve the best accuracy in anomaly detection.
•	CPLE具有灵活的基础模型变更能力，因此可以在异常检测中建立其基本模型，以达到最佳的检测精度。

•	CPLE needs low memory complexity, as opposed to graph-based methods [29] which needs O(n2) memory complexity.

•	CPLE需要低内存复杂性，而不是需要基于O（N2）存储器复杂性的基于图形的方法[29 ]。
•	CPLE is more robust than other semi-supervised learning algorithms because it needs no strong assumptions such as (1) the accurate estimation for data distribution of labeled and unlabeled data (as required by expectation maximization algorithms with generative mixture models [30]) and (2) low density (as required by transductive SVM [31]).
•	·CPLE比其他半监督学习算法更健壮，因为它不需要强的假设，例如（1）标记和未标记数据的数据分布的精确估计（根据生成混合模型的期望最大化算法[30 ]）和（2）低密度（如转导SVM（31）所要求的）。

•	CPLE supports incremental learning. Therefore, when more and more data points are added to a KPI stream, we can continuously train ADS to improve its accuracy.

•	CPLE支持增量学习。因此，当越来越多的数据点被添加到KPI流中时，我们可以不断地训练广告以提高其准确性。

ADS applies CPLE to detect anomalies for KPI streams as follows. For a base-model (a supervised learning based binary classifier) f(x) with parameter vector _, it can be denoted with the form f(x; _) 2 f0; 1g. Then the probability that a basemodel deems a data point as anomalous is g(x; _) = p(f =1jx; _), where g(x; _) 2 [0; 1]. In addition, the negative log loss for binary classifiers takes on the general form: 
ADS应用CPLE检测KPI流的异常，如下所示。对于向量参数为{uy的基模型（基于监督学习的二进制分类器）f（x），它可以用f（x；（u）2f0；1g的形式表示，那么基模型认为数据点异常的概率是g（x；（u）=p（f=1jx；（u），其中g（x；（u）2[0；1]。此外，二进制分类器的负日志丢失呈现一般形式：
 
where N is the number of the data points in the KPI streams of training set, yi is the label of the i-th data point and pi is the i-th discriminative likelihood (DL) [32]. Usually, a machine learning model is aiming to maximize the negative log loss. For the unlabeled data points in the training set (namely, the data points of the newly generated KPI stream), we randomly assign a weight qi to the i-th data point. The objective of CPLE is to minimize the function
其中N是训练集的KPI流中的数据点数量，yi是第i个数据点的标签，pi是第i个判别似然（DL）[32]。通常，机器学习模型的目标是最大化负对数损失。对于训练集中未标记的数据点（即新生成的KPI流的数据点），我们随机将权重qi分配给第i个数据点。CPLE的目标是最小化函数
 
where X is the data set of labeled data points, U is the one of unlabeled data points, and y0 = H(q), where 
其中X是标记数据点的数据集，U是未标记数据点之一，y0=H（q），其中
 

where  

This way, (the parameter vector _ of) the base-model, which serves as the anomaly detection model, is trained based on (X [ U) using actual and hypothesized labels (y [ y0), as well as the weights of data points w, where
这样，作为异常检测模型的基模型的（参数向量）基于（X[U）使用实际和假设的标签（y[y0）以及数据点w的权重进行训练，其中
 
where i 2 f1; :::; jUjg.
In this work, we apply random forest as the base-model of CPLE because of its simplicity, parallelization, and low memory usage, similar to Opprentice [2]. We do not claim the adoption of random forest model as our contribution. 
在那里我2楼1层；：：；jUjg。

在这项工作中，我们使用随机森林作为CPLE的基本模型，因为它简单、并行、低内存使用，类似于oppertice[2]。我们不主张采用随机森林模型作为我们的贡献。

IV. EVALUATION
四、评价
To evaluate the performance of ADS, we have conducted extensive experiments using real-world data collected from a top global online game service. We first introduce the data set (Section IV-A) and metrics (Section IV-B) used for the evaluation experiments. Then, we compare the performance of ADS with that of supervised learning based method such as Opprentice, and unsupervised learning based method including iForest and Donut (Section IV-C). Finally, to highlight the importance of semi-supervised learning, we compare ADS with the combination of ROCKA and Opprentice (Section IV-D).
为了评估广告的性能，我们使用从全球顶级在线游戏服务收集的真实数据进行了广泛的实验。我们首先介绍用于评估实验的数据集（第IV-A节）和度量（第IV-B节）。然后，我们将ADS的性能与基于监督学习的方法（如oppertice）和基于非监督学习的方法（包括iForest和Donut）进行了比较（第4-C节）。最后，为了强调半监督学习的重要性，我们将ADS与ROCKA和Opprentice的组合进行了比较（第4-D节）

A. Data Set
A.数据集
We randomly pick 70 historical KPI streams for clustering and 81 new ones for anomaly detection from a top global online game service. These KPI streams are of the most important three KPIs, including success rate, number of online players and latency. Table II lists the detailed information of the 81 new KPI streams, including the number, interval and length of the KPI streams of each KPI. In addition, it also lists the averaged number and percentage of anomalous data points per KPI stream of each KPI, respectively.
我们从一个顶级的全球在线游戏服务中随机选择70个历史KPI流用于集群，81个新KPI流用于异常检测。这些KPI流是最重要的三个KPI，包括成功率、在线玩家数量和延迟。表二列出了81个新的KPI流的详细信息，包括每个KPI的KPI流的数量、间隔和长度。此外，它还分别列出每个KPI流中异常数据点的平均数量和百分比。

Note that the KPI streams of the same KPI may have very different shapes and thus belong to different clusters. Therefore, experienced operators first manually group the 70 historical KPI streams into five clusters according to their shapes, and then classify the new 81 KPI streams into the five clusters. This serves as the ground truth for clustering.
请注意，同一个KPI的KPI流可能具有非常不同的形状，因此属于不同的集群。因此，经验丰富的操作员首先根据70个历史KPI流的形状将其手动分组为5个集群，然后将新的81个KPI流分类为5个集群。这是聚类的基本事实。

As mentioned in Section III-A, in this work we apply ROCKA to cluster KPI streams. Based on the manual clustering results by operators, we find that ROCKA accurately groups all the 70 historical KPI streams into the right cluster. In addition, ROCKA successfully classifies all the 81 new KPI streams into the correct cluster.
如第III-A节所述，在这项工作中，我们将ROCKA应用于集群KPI流。根据操作员的手动聚类结果，我们发现ROCKA准确地将所有70个历史KPI流分组到正确的集群中。此外，ROCKA成功地将所有81个新的KPI流分类到正确的集群中。
To evaluate the performance of anomaly detection methods, operators also manually label anomalous data points for the
 
81 new KPI streams, as well as the 5 historical KPI streams
that are on cluster centroids (calculated using ROCKA). Note that we focus on anomaly detection on newly emerging KPI streams in this work, and thus only the new 81 KPI streams are used for the following evaluation experiments. Therefore, there is no need to manually labeling the remaining 65 historical KPI streams.
为了评估异常检测方法的性能，操作员还为81个新的KPI流以及位于群集质心上的5个历史KPI流（使用ROCKA计算）手动标记异常数据点。请注意，我们在这项工作中重点关注新出现的KPI流的异常检测，因此只有新的81个KPI流用于以下评估实验因此，不需要手动标记其余65个历史KPI流。

As aforementioned, more than 6000 new KPI streams are produced per 10 days. However, manually clustering and labeling anomalies for thousands of KPI streams is infeasible, considering the long period of KPI streams (one month). Therefore, we randomly selected 151 KPI streams in our evaluation. We believe that the 151 KPI streams are sufficient to evaluate ADS’s performance.
如上所述，每10天产生6000多个新的关键绩效指标流。但是，考虑到KPI流的周期很长（一个月），手动对数千个KPI流的异常进行聚类和标记是不可行的。因此，我们在评估中随机选择了151个KPI流。我们相信151个关键绩效指标流足以评估广告的绩效。

B. Evaluation Metrics
B.评估指标

In real applications, the human operators generally do not care about the point-wise metrics. Therefore, we use a simple strategy following [12]: if any point in an anomaly segment in the ground truth can be detected by a chosen threshold, we say this segment is detected correctly, and all points in this segment are treated as if they can be detected by this threshold. Meanwhile, the points outside the anomaly segments are treated as usual. The precision, recall, F-score and best F-score are then computed accordingly.
在实际应用中，人工操作人员通常不关心点度量。因此，我们使用了一个简单的策略，如下[12]：如果地面真值异常段中的任何一点可以通过选择的阈值检测到，我们说该段被正确检测到，并且该段中的所有点都被视为可以通过该阈值检测到。同时，对异常段外的点按常规处理。然后相应地计算精确性、召回率、F得分和最佳F得分。
As is with [12], we apply the best F-score as the metric to evaluate anomaly detection methods. The best F-score indicates the best possible performance of an anomaly detection method on a particular testing set, given an optimal global threshold. In practice, the best F-score is mostly consistent with Area Under the (ROC) Curve (AUC).
与文献[12]一样，我们将最佳F值作为评价异常检测方法的指标。在给定一个最优全局阈值的情况下，最佳F值表示异常检测方法在特定测试集上的最佳性能。在实践中，最佳F值与ROC曲线下面积（AUC）基本一致。

Note that in Section IV-C and Section IV-D, we tune aThld based on the labels of the testing set (namely, the back 40% of every new KPI stream) to obtain the best F-score. Although it is not entirely practical in real-world applications, it fully compares the best performance of ADS, iForest, Donut, Opprentice and the combination of ROCKA and Opprentice.
注：在第四节C和第四节D中，我们根据测试集标签（“NAMELY”），在每一新KPI流域的40%的基础上，调谐ATHELD，以达到最佳F-SCORE。虽然这在现实世界中并不是完全的实际应用，但它完全比较了ADS、IFOREST、DONUT、Opprentice和Rocka和Opprentice的最佳性能。

C. Evaluation of The Overall Performance
C.总体性能评价
To evaluate the performance of ADS in anomaly detection for KPI streams, we calculate its best F-score, and compare it with that of iForest [11], Donut [12] and Opprentice [2].
为了评价ADS在KPI流异常检测中的性能，我们计算了其最佳F值，并与iForest[11]、Donut[12]和Opprentice[2]进行了比较。

For each of the 81 new KPI streams, we train ADS using the features extracted from the front 60% of it, as well as the features and manual labels of its cluster centroid (KPI stream). Then we use this ADS model to detect anomalies on the back 40% of this new KPI stream. As for iForest, Donut and Opprentice, we divide each new KPI stream (of the 81 KPI streams) into training set and testing set, whose ratios are (front) 60% and (back) 40%, respectively.
对于81个新的KPI流中的每一个，我们使用从它前面提取的60%的特征，以及它的簇质心（KPI流）的特征和手动标签来训练广告。然后我们使用这个ADS模型来检测这个新KPI流后面40%的异常。对于iForest、Donut和oppertice，我们将每个新的KPI流（81个KPI流中的一个）分为训练集和测试集，其比率分别为（前）60%和（后）40%。
 
Fig. 4: CDFs of the best F-scores of each new KPI stream using ADS, iForest, Donut and Opprentice, respectively.
图4：分别使用ADS、iForest、Donut和Oppertice的每个新的关键绩效指标流的最佳F分数的CDF。

Figure 4 shows the cumulative distribution functions (CDFs) of the best F-scores of each KPI stream (of the new 81 KPI streams) using the above four methods. Note that in this CDF figure, being closer to the upper left means worse performance, and being closer to the bottom right means better performance. We can see that both ADS and Opprentice perform superior in detecting anomalies for KPI streams, with 80% of best Fscores over 0.8. Their performance is much better than that of iForest and Donut, for the following reasons: (1) iForest is sensitive to noises in the training set because it does not utilize any labels [11]. (2) Donut is a deep Bayesian model, which requires a lot of training data to get good results (say six months worth of KPI streams) [12]. However, as Table II shows, we have to train the model using 60% of one month, namely 18 days, worth of newly emerging KPI streams, which is a too small amount of training data for Donut.
图4显示了使用上述四种方法的每个KPI流（新的81个KPI流）的最佳F分数的累积分布函数（CDFs）。注意，在这个CDF图中，靠近左上角意味着性能更差，而靠近右下角意味着性能更好。我们可以看到，ADS和oppertice在检测KPI流异常方面都表现出色，80%的最佳fscore超过0.8。他们的表现比iForest和Donut要好得多，原因如下：（1）iForest对训练集中的噪音很敏感，因为它没有使用任何标签[11]。（2）甜甜圈是一个深度贝叶斯模型，需要大量的训练数据才能获得良好的效果（比如说6个月的KPI流）[12]。然而，如表二所示，我们必须使用一个月的60%即18天的新出现的KPI流来训练模型，这对于甜甜圈来说是太少的训练数据。

To intuitively compare the best F-scores of ADS, iForest, Opprentice and Donut, we list the average best F-scores of the above four methods on the five clusters in TABLE III, respectively. ADS and Opprentice perform well across all the five clusters and much better than iForst and Donut, demonstrating that it is important to utilize anomaly labels in anomaly detection for newly emerging KPI streams. Specifically, ADS improves the average best F-score by 61.40% (as opposed to Donut) to 360% (as opposed to iForest).
为了直观地比较广告、iForest、Oppentice和Donut的最佳F得分，我们将上述四种方法的平均最佳F得分分别列在表三的五个聚类上。ADS和oppentice在所有五个集群中都表现良好，比iForst和Donut要好得多，这表明在异常检测中使用异常标签对于新出现的KPI流是非常重要的。具体来说，广告将平均最佳F得分提高了61.40%（相对于甜甜圈）至360%（相对于iForest）。

Although the supervised learning based method, Opprentice, performs similarly to ADS, it needs much more labeling works. For example, in this setting, ADS is trained based on only five labeled KPI streams on cluster centroids, while Opprentice is trained using all the 81 labeled KPI streams. As aforementioned, over 6000 new KPI streams are produced per 10 days in the studied online gaming service. Manually labeling all the newly emerging KPI streams to detect KPI anomalies is infeasible in practice. Consequently, Opprentice is not appropriate for our scenario.
虽然基于监督学习的方法opperentice的性能与ADS类似，但它需要更多的标记工作。例如，在此设置中，ADS仅基于群集质心上的五个标记的KPI流进行训练，而Opprentice则使用所有81个标记的KPI流进行训练。如上所述，在所研究的在线游戏服务中，每10天产生6000多个新的关键绩效指标流。手动标记所有新出现的KPI流以检测KPI异常在实践中是不可行的。因此，机会不适合我们的情况。

 


 
D. Evaluation of CPLE
D.CPLE评估

To the best of our knowledge, this is the first work to apply semi-supervised learning to the KPI anomaly detection problem. We adopt a robust semi-supervised learning model, CPLE, which is suitable for KPI anomaly detection and only requires similar (not necessarily the same) data distribution between the existing labeled KPI stream and the new KPI stream.
据我们所知，这是第一个将半监督学习应用于KPI异常检测问题的工作。我们采用一种健壮的半监督学习模型CPLE，它适合于KPI异常检测，并且只需要在现有的标记KPI流和新的KPI流之间相似（不一定相同）的数据分布。

To evaluate the performance of CPLE, we compare the performance of ADS, which is the combination of ROCKA and CPLE, to that of the combination of ROCKA and a stateof- art supervised learning method – Opprentice [2] (ROCKA + Opprentice henceforth). We set up ROCKA + Opprentice as follows. We first apply ROCKA to group the 70 historical KPI streams into five clusters, and classify the 81 new KPI streams into these clusters. For each cluster, we train Opprentice using the features and manual labels of its centroid KPI streams. After that, we detect anomalies for the back 40% of each new KPI stream using the Opprentice model trained based on this new KPI stream’s cluster centroid.
为了评估CPLE的性能，我们将ROCKA和CPLE相结合的自我训练异常检测的性能与ROCKA和一种先进的监督学习方法oppentice[2]相结合的自我训练异常检测的性能进行了比较（ROCKA+oppentice从此）。我们建立了ROCKA+Oppertice，如下所示。我们首先应用ROCKA将70个历史KPI流分组为5个集群，并将81个新KPI流分类为这些集群。对于每个集群，我们使用其质心KPI流的特性和手动标签来训练oppertice。之后，我们使用基于新的KPI流的聚类中心训练的oppertice模型检测每个新KPI流后面40%的异常。

TABLE III compares the average best F-scores of ADS and ROCKA + Opprentice on each cluster. We can see that ADS outperforms ROCKA + Opprentice on every cluster, and greatly outperforms it by 35.82% on cluster A. TABLE IV lists the new KPI streams where ADS performs significantly better than ROCKA + Opprentice, and the best F-scores of the above two methods on these KPI streams, respectively.
表三比较了自我训练异常检测和ROCKA+Oppertice在每个集群中的平均最佳F得分。我们可以看到，在每个集群上，自我训练异常检测的表现都优于ROCKA+Opprentice，在集群A上，自我训练异常检测的表现大大优于ROCKA+Opprentice的35.82%。表四列出了自我训练异常检测表现明显优于ROCKA+Opprentice的新的关键绩效指标流，以及上述两种方法在这些关键绩效指标流上的最佳F分数。
 

Fig. 5: The anomaly detection results of ROCKA + Opprentice on KPI stream , and ’s cluster centroid KPI stream. The red data points are anomalous determined by ROCKA + Opprentice while in actual they are normal.

图5:ROCKA+Oppertice对KPI流和其聚类中心KPI流的异常检测结果。红色数据点是由ROCKA+oppertice异常确定的，而实际上是正常的。

Here we explain why ADS performs better than ROCKA + Opprentice. KPI stream clustering methods such as ROCKA usually extract baselines (namely underlying shapes) from KPI streams and ignore fluctuations. However, the fluctuations of KPI streams can impact anomaly detection. For example, Figure 5 shows the new KPI stream , and the KPI stream on the its cluster centroid.  KPI stream   and the centroid KPI stream have very similar baselines, but they have different fluctuation degrees, which is not uncommon in practice. This lead to that ROCKA + Opprentice, which is trained based only on the centroid KPI stream, generates a lot of false alarms.
这里我们解释为什么自我训练异常检测比ROCKA+Opprentice表现更好。像ROCKA这样的KPI流聚类方法通常从KPI流中提取基线（即底层形状）并忽略波动。然而，KPI流的波动会影响异常检测。例如，图5显示了新的KPI流，以及其集群质心上的KPI流。KPI流和质心KPI流具有非常相似的基线，但它们具有不同的波动程度，这在实践中并不少见。这就导致了ROCKA+Opprentice，它只基于质心KPI流进行训练，会产生很多错误警报。

ADS addresses the above problem effectively using semisupervised learning. In other words, it learns not only from the labels of the centroid KPI stream, but also from the fluctuation degree of the new KPI stream. This is consistent with the observation that the model trained based on both labeled and unlabeled data should not be worse than the one trained based only on the labeled data [28]. 

自我训练异常检测利用半监督学习有效地解决了上述问题。换句话说，它不仅从质心KPI流的标签中学习，而且还从新KPI流的波动程度中学习。这与基于标记数据和未标记数据训练的模型不应比仅基于标记数据训练的模型差的观察结果一致[28]。

The experiment results strongly demonstrate ADS’s robustness in KPI anomaly detection.
实验结果有力地证明了自我训练异常检测在KPI异常检测中的稳健性。

V. CONCLUSION
V. 结论

To the best of our knowledge, this paper is the first to identify the common and important problem of rapid deployment of anomaly detection models for large number of emerging KPI streams, without manual algorithm selection, parameter tuning, or new anomaly labeling for any newly generated KPI streams. We propose the first framework ADS that tackles this problem via clustering and semi-supervised learning, which is the first time that semi-supervised learning is applied to KPI anomaly detection. Our extensive experiments using real world data show that, with the labels of only the 5 cluster centroids of 70 historical KPI streams, ADS achieves an averaged best F-score of 0.92 on 81 new KPI streams, almost the same as the state-of-art supervised approach [2], and greatly outperforms an unsupervised approach Isolation Forest [14] by 360% and the state-of-art unsupervised approach Donut [12] by 61.40% on average.
据我们所知，本文首先确定了快速部署大量新出现的KPI流异常检测模型的常见和重要问题，而不需要对任何新生成的KPI流进行手动算法选择、参数调整或新的异常标记。我们提出了第一个通过聚类和半监督学习来解决这一问题的框架自我训练异常检测，这是半监督学习首次应用于KPI异常检测。我们使用真实世界数据进行的大量实验表明，在70个历史KPI流的5个簇质心的标记下，自我训练异常检测在81个新KPI流上的平均最佳F得分为0.92，几乎与最先进的监督方法[2]相同，并且大大优于无监督方法隔离林[14]360%和最先进的无监督方法甜甜圈平均增长61.40%。

We believe that ADS is a significant step towards practical anomaly detection on large-scale KPI streams in Internetbased services. In the future, we plan to adopt more advanced techniques (e.g. transfer learning [33]) to further improve ADS’s performance.

我们认为，自我训练异常检测是在基于Internet的服务中实现大规模关键绩效指标流异常检测的重要一步在未来，我们计划采用更先进的技术（如转移学习[33]），以进一步提高自我训练异常检测的性能。

VI. ACKNOWLEDGEMENTS
We thank Shuai Yang, Tianheng Zuo for their helpful suggestions.
The work was supported by National Natural Science Foundation of China (NSFC) under grant No. 61402257, No.61472214, No. 61472210 and No. 61772307, and Tsinghua-Tencent Joint Laboratory for Internet Innovation Technology.








REFERENCES
[1] Y. Chen, R. Mahajan, B. Sridharan, and Z.-L. Zhang, “A providerside view of web search response time,” in ACM SIGCOMM Computer Communication Review, vol. 43, no. 4. ACM, 2013, pp. 243–254.
[2] D. Liu, Y. Zhao, H. Xu, Y. Sun, D. Pei, J. Luo, X. Jing, and M. Feng,
“Opprentice: towards practical and automatic anomaly detection through machine learning,” in Proceedings of the 2015 ACM Conference on Internet Measurement Conference. ACM, 2015, pp. 211–224.
[3] S. Zhang, Y. Liu, D. Pei, Y. Chen, X. Qu, S. Tao, and Z. Zang, “Rapid and robust impact assessment of software changes in large internetbased services,” in International Conference on emerging Networking EXperiments and Technologies (CoNEXT), 2015, pp. 1–13.
[4] M. Ma, S. Zhang, D. Pei, X. Huang, and H. Dai, “Robust and rapid adaption for concept drift in software system anomaly detection,” in IEEE International Symposium on Software Reliability Engineering (ISSRE), 2018.
[5] Y. Sun, Y. Zhao, Y. Su, D. Liu, X. Nie, Y. Meng, S. Cheng, D. Pei,
S. Zhang, X. Qu et al., “Hotspot: Anomaly localization for additive
kpis with multi-dimensional attributes,” IEEE Access, vol. 6, pp. 10 909–10 923, 2018.
[6] A. Mahimkar, Z. Ge, J. Wang, J. Yates, Y. Zhang, J. Emmons,
B. Huntley, and M. Stockert, “Rapid detection of maintenance induced changes in service performance,” in Proceedings of the Seventh COnference on Emerging Networking EXperiments and Technologies, ser. CoNEXT ’11. New York, NY, USA: ACM, 2011, pp. 13:1–13:12. 
[Online]. Available: http://doi.acm.org/10.1145/2079296.2079309
[7] P. Barford, J. Kline, D. Plonka, and A. Ron, “A signal analysis of
network traffic anomalies,” in Proceedings of the 2nd ACM SIGCOMM Workshop on Internet measurment. ACM, 2002, pp. 71–82.
[8] Y. Zhang, Z. Ge, A. Greenberg, and M. Roughan, “Network
anomography,” in Proceedings of the 5th ACM SIGCOMM Conference on Internet Measurement, ser. IMC ’05. Berkeley, CA, USA: USENIX Association, 2005, pp. 30–30. [Online]. Available: http://dl.acm.org/citation.cfm?id=1251086.1251116
[9] H. Yan, A. Flavel, Z. Ge, A. Gerber, D. Massey, C. Papadopoulos,
H. Shah, and J. Yates, “Argus: End-to-end service anomaly detection and localization from an isp’s point of view,” in INFOCOM, 2012 Proceedings IEEE. IEEE, 2012, pp. 2756–2760.
[10] N. Laptev, S. Amizadeh, and I. Flint, “Generic and scalable framework for automated time-series anomaly detection,” in Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2015, pp. 1939–1947.
[11] Z. Ding and M. Fei, “An anomaly detection approach based on isolation forest algorithm for streaming data using sliding window,” IFAC Proceedings Volumes, vol. 46, no. 20, pp. 12–17, 2013.
[12] H. Xu, W. Chen, N. Zhao, Z. Li, J. Bu, Z. Li, Y. Liu, Y. Zhao, D. Pei, Y. Feng et al., “Unsupervised anomaly detection via variational autoencoder for seasonal kpis in web applications,” in Proceedings of the 2018 World Wide Web Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2018, pp. 187–196.
[13] S. Zhang, Y. Liu, D. Pei, Y. Chen, X. Qu, S. Tao, Z. Zang, X. Jing, and M. Feng, “Funnel: Assessing software changes in web-based services,” IEEE Transactions on Service Computing, 2016.
[14] Y.-L. Zhang, L. Li, J. Zhou, X. Li, and Z.-H. Zhou, “Anomaly detection with partially observed anomalies,” in Companion of the The Web Conference 2018 on The Web Conference 2018. International World Wide Web Conferences Steering Committee, 2018, pp. 639–646.
[15] Z. Li, Y. Zhao, R. Liu, and D. Pei, “Robust and rapid clustering of kpis for large-scale anomaly detection,” Quality of Service (IWQoS), pp. 1–10, 2018.
[16] O. Chapelle, B. Scholkopf, and A. Zien, “Semi-supervised learning (chapelle, o. et al., eds.; 2006)[book reviews],” IEEE Transactions on Neural Networks, vol. 20, no. 3, pp. 542–542, 2009.
[17] W. Meng, Y. Liu, S. Zhang, D. Pei, H. Dong, L. Song, and X. Luo,
“Device-agnostic log anomaly classification with partial labels,” Quality of Service (IWQoS), pp. 1–10, 2018.
[18] S. Zhang, Y. Liu, W. Meng, Z. Luo, J. Bu, S. Yang, P. Liang, D. Pei, J. Xu, Y. Zhang et al., “Prefix: Switch failure prediction in datacenter networks,” Proceedings of the ACM on Measurement and Analysis of Computing Systems, vol. 2, no. 1, p. 2, 2018.
[19] S. Zhang, W. Meng, J. Bu, S. Yang, Y. Liu, D. Pei, J. Xu, Y. Chen,
H. Dong, X. Qu et al., “Syslog processing for switch failure diagnosis and prediction in datacenter networks,” in Quality of Service (IWQoS), 2017 IEEE/ACM 25th International Symposium on. IEEE, 2017,pp. 1–10.
[20] C. Rosenberg, M. Hebert, and H. Schneiderman, “Semi-supervised selftraining of object detection models,” 2005.
[21] R. A. R. Ashfaq, X.-Z. Wang, J. Z. Huang, H. Abbas, and Y.-L.
He, “Fuzziness based semi-supervised learning approach for intrusion detection system,” Information Sciences, vol. 378, pp. 484–497, 2017.
[22] K. Noto, C. Brodley, and D. Slonim, “Frac: a feature-modeling approach for semi-supervised and unsupervised anomaly detection,” Data mining and knowledge discovery, vol. 25, no. 1, pp. 109–133, 2012.
[23] T. W. Liao, “Clustering of time series dataa survey,” Pattern recognition, vol. 38, no. 11, pp. 1857–1874, 2005.
[24] S.-B. Lee, D. Pei, M. Hajiaghayi, I. Pefkianakis, S. Lu, H. Yan, Z. Ge, J. Yates, and M. Kosseifi, “Threshold compression for 3g scalable monitoring,” in INFOCOM, 2012 Proceedings IEEE. IEEE, 2012, pp.1350–1358.
[25] “Amazon cloudwatch alarm,” http://docs.aws.amazon.com/
AmazonCloudWatch/latest/DeveloperGuide/ConsoleAlarms.html.
[26] D. R. Choffnes, F. E. Bustamante, and Z. Ge, “Crowdsourcing servicelevel network event monitoring,” in Proceedings of the ACM SIGCOMM 2010 Conference, ser. SIGCOMM ’10. ACM, 2010, pp. 387–398. [Online]. Available: http://doi.acm.org/10.1145/1851182.1851228
[27] B. Krishnamurthy, S. Sen, Y. Zhang, and Y. Chen, “Sketch-based change detection: methods, evaluation, and applications,” in Proceedings of the 3rd ACM SIGCOMM conference on Internet measurement. ACM, 2003, pp. 234–247.
[28] M. Loog, “Contrastive pessimistic likelihood estimation for semisupervised classification,” IEEE transactions on pattern analysis and machine intelligence, vol. 38, no. 3, pp. 462–475, 2016. [29] G. Camps-Valls, T. V. B. Marsheva, and D. Zhou, “Semi-supervised graph-based hyperspectral image classification,” IEEE Transactions on Geoscience and Remote Sensing, vol. 45, no. 10, pp. 3044–3054, 2007.
[30] K. Nigam, A. K. McCallum, S. Thrun, and T. Mitchell, “Text classification from labeled and unlabeled documents using em,” Machine learning, vol. 39, no. 2-3, pp. 103–134, 2000.
[31] T. Joachims, “Transductive inference for text classification using support vector machines,” in ICML, vol. 99, 1999, pp. 200–209.
[32] A. Y. Ng and M. I. Jordan, “On discriminative vs. generative classifiers: A comparison of logistic regression and naive bayes,” in Advances in neural information processing systems, 2002, pp. 841–848.
[33] S. J. Pan, Q. Yang et al., “A survey on transfer learning,” IEEE
Transactions on knowledge and data engineering, vol. 22, no. 10, pp.1345–1359, 2010.
