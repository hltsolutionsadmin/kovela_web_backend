using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace DeepFace.Models
{
    public class FaceEntity
    {
        [Key, DatabaseGenerated(DatabaseGeneratedOption.Identity)]
        public int FaceId { get; set; }

        [Required]
        public string ExternalId { get; set; } = default!; // GUID we pass to Python /enroll and store as image name

        [Required]
        public string DescriptorJson { get; set; } = default!; // embedding as JSON array of floats

        public string? ThumbnailBase64 { get; set; } // optional thumbnail (returned by Python)

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    }
}
