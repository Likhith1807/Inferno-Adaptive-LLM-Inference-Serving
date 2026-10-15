output "public_ip" {
  value = aws_instance.gpu.public_ip
}

output "instance_id" {
  value = aws_instance.gpu.id
}

output "ssh_command" {
  value = "ssh -i <your-key.pem> ubuntu@${aws_instance.gpu.public_ip}"
}
