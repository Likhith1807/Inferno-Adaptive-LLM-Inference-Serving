# Multi-GPU benchmark instance for the cloud phase (CO3).
#
# STATUS: written but not applied in the authoring session (no cloud credentials there). Run
# `terraform validate` and `terraform plan` before `apply`, and read the cost note in README.md.
#
# Cost guard: the instance terminates itself after var.max_hours regardless of what the orchestration
# script does (instance_initiated_shutdown_behavior = "terminate" + a timed shutdown in user_data).

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# AWS Deep Learning AMI: NVIDIA driver, CUDA and Docker preinstalled.
data "aws_ami" "dlami" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.* (Ubuntu 22.04) *"]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

resource "aws_security_group" "ssh" {
  name_prefix = "inferno-ssh-"
  description = "SSH from the operator only"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_cidr]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "gpu" {
  ami                                  = data.aws_ami.dlami.id
  instance_type                        = var.instance_type
  key_name                             = var.key_name
  vpc_security_group_ids               = [aws_security_group.ssh.id]
  instance_initiated_shutdown_behavior = "terminate"

  root_block_device {
    volume_size = var.disk_gb
    volume_type = "gp3"
  }

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        spot_instance_type = "one-time"
      }
    }
  }

  user_data = <<-EOT
    #!/bin/bash
    # Hard cost ceiling: power off (=> terminate) after max_hours no matter what.
    shutdown -h +${var.max_hours * 60}
  EOT

  tags = {
    Name    = "inferno-bench"
    Project = "inferno"
  }
}
