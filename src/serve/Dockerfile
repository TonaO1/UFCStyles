# Pulls from AWS base ECR image and builds atop it
FROM public.ecr.aws/lambda/python:3.12 

# Install dependencies (minimal: numpy and boto3 only)
RUN pip install -r requirements.txt "${LAMBDA_TASK_ROOT}"

# Copy handler and weights
COPY src/serve/handler.py ${LAMBDA_TASK_ROOT}/
COPY data/models/contrastive_style_8/serving/fight_model.npz ${LAMBDA_TASK_ROOT}/
COPY src/serve/inference.py ${LAMBDA_TASK_ROOT}/
COPY src/serve/requirements.txt ${LAMBDA_TASK_ROOT}/


# Set the Lambda handler
CMD [ "handler.lambda_handler" ]
