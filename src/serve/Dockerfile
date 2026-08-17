FROM public.ecr.aws/lambda/python:3.12

# Copy handler and weights
COPY src/serve/handler.py ${LAMBDA_TASK_ROOT}/
COPY data/models/encoder_weights.npz ${LAMBDA_TASK_ROOT}/
COPY data/snapshots/scaler.pkl ${LAMBDA_TASK_ROOT}/

# Install dependencies (minimal: numpy and boto3 only)
RUN pip install numpy boto3 --target "${LAMBDA_TASK_ROOT}"

# Set the Lambda handler
CMD [ "handler.lambda_handler" ]
