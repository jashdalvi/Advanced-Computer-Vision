	?i2?ma1@?i2?ma1@!?i2?ma1@	???4eK?????4eK??!???4eK??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?i2?ma1@?Nt	??A|?w*1@Y??Ia????*	?E???|U@2F
Iterator::Modela\:?<??!??U?w?E@)???@???1?~?u??7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??i?:??!?o???e;@)O??e???1o??>16@:Preprocessing2U
Iterator::Model::ParallelMapV2$??ŋ???!???J?3@)$??ŋ???1???J?3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateѕT? ??!?`l???4@)??V	???1?-?U?/&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?j??躀?!|?c#@)?j??躀?1|?c#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?Y??!Ũ?!uQ?_?$L@)?AA)Z?w?1????J?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???:TSr?!p??'?@)???:TSr?1p??'?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?e???~??!8N?s9&6@)? ݗ3?U?1O??w	???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???4eK??Ip˚??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Nt	???Nt	??!?Nt	??      ??!       "      ??!       *      ??!       2	|?w*1@|?w*1@!|?w*1@:      ??!       B      ??!       J	??Ia??????Ia????!??Ia????R      ??!       Z	??Ia??????Ia????!??Ia????b      ??!       JCPU_ONLYY???4eK??b qp˚??X@