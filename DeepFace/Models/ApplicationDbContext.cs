using Microsoft.EntityFrameworkCore;

namespace DeepFace.Models
{
    public class ApplicationDbContext : DbContext
    {
        public ApplicationDbContext(DbContextOptions<ApplicationDbContext> options) : base(options) { }

        public DbSet<FaceEntity> Faces { get; set; }
    }
}
