using Microsoft.EntityFrameworkCore;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace DeepFace.Models
{
    public class FaceEntity
  {
    [Key, DatabaseGenerated(DatabaseGeneratedOption.Identity)]
    public int FaceId { get; set; }

    [Required, MaxLength(100)]
    public string ExternalId { get; set; } = default!;

    [Required]
    public string DescriptorJson { get; set; } = default!; // embedding as JSON array of floats

    public string? ThumbnailBase64 { get; set; }

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    public bool Consent { get; set; }
  }
}
