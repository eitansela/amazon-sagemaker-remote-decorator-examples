import boto3
import sagemaker

from sagemaker.remote_function import remote

sm_session = sagemaker.Session(boto_session=boto3.session.Session(region_name="us-east-1"))
settings = dict(
    sagemaker_session=sm_session,
    role="AmazonSageMaker-ExecutionRole-20190829T190746", # REPLACE WITH YOUR OWN ROLE HERE
    instance_type="ml.m5.xlarge",
    dependencies='./requirements.txt',
)


@remote(**settings)
def divide(x, y):
    print(f"Calculating {x}/{y}")
    return x / y


if __name__ == "__main__":
    print(divide(2, 3.0))
