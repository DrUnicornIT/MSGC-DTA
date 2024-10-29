import kfp
from kfp import dsl
from kfp.dsl import PipelineVolume

def cleaningdata_op(image):
    op = dsl.ContainerOp(
        name='Clean data',
        image=image,
        command=['python3', 'main.py', ' > out.txt'],
    )
    op.container.set_image_pull_policy('Always')
    op.container.set_cpu_request('12000m').set_cpu_limit(
        '12000m').set_memory_request('16Gi').set_memory_limit('16Gi').set_gpu_limit(1)
    return op

@dsl.pipeline(
    name="Cleaning data",
    description='Cleaning data'
)

def my_pipeline(image):
    preprocessing = cleaningdata_op(image)

from kubernetes import client as k8s_client

pipeline_conf = kfp.dsl.PipelineConf()
pipeline_conf.set_image_pull_secrets([k8s_client.V1ObjectReference(name="dockerhub-private")])

kfp.compiler.Compiler().compile(my_pipeline, 'pipeline_training.yaml', pipeline_conf=pipeline_conf)