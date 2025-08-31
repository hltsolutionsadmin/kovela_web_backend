using System.ComponentModel.DataAnnotations;

namespace DeepFace.Models
{
    public class UserDetails
    {
        [Key]
        public int Id { get; set; }
        public int FaceId { get; set; }
        public string? Name { get; set; }
        public string? PhoneNumber { get; set; }
    }
}
