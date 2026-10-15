variable "region" {
  description = "AWS region (check GPU instance availability and quotas there first)"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "GPU instance type. g5.12xlarge = 4x A10G 24GB. p4d.24xlarge = 8x A100 40GB (much more expensive)."
  type        = string
  default     = "g5.12xlarge"
}

variable "key_name" {
  description = "Name of an existing EC2 key pair for SSH"
  type        = string
}

variable "ssh_cidr" {
  description = "CIDR allowed to SSH in, e.g. \"203.0.113.7/32\". Never 0.0.0.0/0."
  type        = string
  validation {
    condition     = var.ssh_cidr != "0.0.0.0/0"
    error_message = "ssh_cidr must not be open to the whole internet."
  }
}

variable "max_hours" {
  description = "Hard lifetime of the instance in hours (self-terminates)"
  type        = number
  default     = 6
}

variable "use_spot" {
  description = "Use a spot instance (cheaper; can be reclaimed mid-run)"
  type        = bool
  default     = false
}

variable "disk_gb" {
  type    = number
  default = 200
}
