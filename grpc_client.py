import grpc
import base64
import logging
import os
import facial_recognition_pb2
import facial_recognition_pb2_grpc


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gRPC Client")


def read_image_as_base64(image_path):
    """Read an image file and encode it as Base64."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read())  # Base64 encode the image data as bytes


def save_image(user_id, image_path):
    """Call the SaveImage RPC."""
    try:
        image_data = read_image_as_base64(image_path)
        request = facial_recognition_pb2.SaveImageRequest(user_id=str(user_id), image=image_data)

        server_address = os.getenv("GRPC_SERVER", "localhost:50051")
        with grpc.insecure_channel(server_address) as channel:
            stub = facial_recognition_pb2_grpc.FacialRecognitionServiceStub(channel)
            response = stub.SaveImage(request)

        logger.info(f"SaveImage Response: Success={response.success}, Message='{response.message}'")
        return response
    except Exception as e:
        logger.error(f"Error in save_image: {e}")


def get_user_id(image_path):
    """Call the GetUserId RPC."""
    try:
        image_data = read_image_as_base64(image_path)
        request = facial_recognition_pb2.GetUserIdRequest(image=image_data)

        server_address = os.getenv("GRPC_SERVER", "localhost:50051")
        with grpc.insecure_channel(server_address) as channel:
            stub = facial_recognition_pb2_grpc.FacialRecognitionServiceStub(channel)
            response = stub.GetUserId(request)

        logger.info(f"GetUserId Response: Success={response.success}, Message='{response.message}', UserID={response.user_id}")
        return response
    except Exception as e:
        logger.error(f"Error in get_user_id: {e}")


if __name__ == "__main__":
    image_path = input("Enter the image path: ")
    user_id = input("Enter the user ID: ")
    save_image(user_id, image_path)

    identification_path = input("Enter the identification image path: ")
    get_user_id(identification_path)
