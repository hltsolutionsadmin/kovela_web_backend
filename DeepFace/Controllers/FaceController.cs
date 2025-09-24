using DeepFace.Models;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using System.Text.Json;

namespace DeepFace.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class FaceController : ControllerBase
    {
        private readonly ApplicationDbContext _context;
        private static readonly HttpClient _http = new HttpClient
        {
            BaseAddress = new Uri("http://192.168.1.7:8000/") // Flask server base URL
        };

        private static readonly JsonSerializerOptions JsonOpts = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };

        public FaceController(ApplicationDbContext context)
        {
            _context = context;
        }

        // DTOs
        public class B64Request { public string Base64Image { get; set; } = default!; public bool? Consent { get; set; } }

        public class UserDetailsRequest { public int FaceId { get; set; } public string? Name { get; set; } public string? PhoneNumber { get; set; } }
        public class EnrollRequestPython { public string Base64Image { get; set; } = default!; public string ExternalId { get; set; } = default!; }
        public class EnrollResponsePython { public string? Error { get; set; } public string? Message { get; set; } public string? ExternalId { get; set; } public float[]? Embedding { get; set; } public string? Thumbnail { get; set; } public int? Count { get; set; } }
        public class CheckRequestPython { public string Base64Image { get; set; } = default!; public int TopK { get; set; } = 5; public double Threshold { get; set; } = 0.35; public bool IncludeThumbnails { get; set; } = true; }
        public class MatchItem { public string ExternalId { get; set; } = default!; public double Score { get; set; } public string? Thumbnail { get; set; } public int? FaceId { get; set; } }
        public class CheckResponsePython { public string? Error { get; set; } public string? Type { get; set; } public double BestScore { get; set; } public double Threshold { get; set; } public List<MatchItem>? Matches { get; set; } }

        [HttpPost("check")]
        public async Task<IActionResult> CheckFace([FromBody] B64Request request)
        {
            if (string.IsNullOrWhiteSpace(request.Base64Image) || request.Base64Image.Length < 1000)
                return BadRequest("Invalid Base64Image.");

            try
            {
                var pyReq = new CheckRequestPython
                {
                    Base64Image = request.Base64Image,
                    TopK = 5,
                    Threshold = 0.35,
                    IncludeThumbnails = true
                };

                using var resp = await _http.PostAsJsonAsync("check", pyReq, JsonOpts);
                var body = await resp.Content.ReadAsStringAsync();

                if (!resp.IsSuccessStatusCode)
                    return StatusCode((int)resp.StatusCode, new { error = $"Flask error: {body}" });

                CheckResponsePython? py = null;
                try { py = JsonSerializer.Deserialize<CheckResponsePython>(body, JsonOpts); }
                catch (Exception ex) { return StatusCode(500, new { error = $"JSON parse error: {ex.Message}", raw = body }); }

                if (py == null)
                    return StatusCode(502, new { error = "Python service returned no data", raw = body });
                if (!string.IsNullOrEmpty(py.Error))
                    return BadRequest(new { error = py.Error });

                const double minScore = 0.5;
                if (py.Matches != null)
                {
                    py.Matches = py.Matches
                        .Where(m => m.Score >= minScore)
                        .OrderByDescending(m => m.Score)
                        .Take(5)
                        .ToList();

                    var extIds = py.Matches.Select(m => m.ExternalId).Distinct().ToList();
                    var map = _context.Faces
                       .Where(f => extIds.Contains(f.ExternalId))
                       .ToDictionary(f => f.ExternalId, f => f.FaceId);

                    foreach (var m in py.Matches)
                       if (map.TryGetValue(m.ExternalId, out var faceId)) m.FaceId = faceId;
                }

                return Ok(py);
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { error = $"Face check failed: {ex.Message}" });
            }
        }

        [HttpPost("enroll")]
        public async Task<IActionResult> EnrollFace([FromBody] B64Request request)
        {
            if (string.IsNullOrWhiteSpace(request.Base64Image) || request.Base64Image.Length < 1000)
                return BadRequest("Invalid Base64Image.");

            try
            {
                string externalId = Guid.NewGuid().ToString();
                var pyReq = new EnrollRequestPython { Base64Image = request.Base64Image, ExternalId = externalId };

                using var resp = await _http.PostAsJsonAsync("enroll", pyReq, JsonOpts);
                var body = await resp.Content.ReadAsStringAsync();

                if (!resp.IsSuccessStatusCode)
                    return StatusCode((int)resp.StatusCode, new { error = $"Flask error: {body}" });

                EnrollResponsePython? py = null;
                try { py = JsonSerializer.Deserialize<EnrollResponsePython>(body, JsonOpts); }
                catch (Exception ex) { return StatusCode(500, new { error = $"JSON parse error: {ex.Message}", raw = body }); }

                if (py == null)
                    return StatusCode(502, new { error = "Python service returned no data", raw = body });
                if (!string.IsNullOrEmpty(py.Error))
                    return BadRequest(new { error = py.Error });

                var entity = new FaceEntity
                {
                    ExternalId = externalId,
                    DescriptorJson = JsonSerializer.Serialize(py.Embedding ?? Array.Empty<float>()),
                    ThumbnailBase64 = py.Thumbnail,
                    Consent = true
                };
                _context.Faces.Add(entity);
                await _context.SaveChangesAsync();

                return Ok(new { message = "User enrolled successfully", faceId = entity.FaceId, externalId, consent = entity.Consent });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { error = $"Face enrollment failed: {ex.Message}" });
            }
        }

        [HttpPost("storeUserDetails")]
        public async Task<IActionResult> StoreUserDetails([FromBody] UserDetailsRequest request)
        {
            if (request.FaceId <= 0)
                return BadRequest("Valid FaceId is required.");

            try
            {
                // Validate that FaceId exists in Faces table
                var faceExists = await _context.Faces.AnyAsync(f => f.FaceId == request.FaceId);
                if (!faceExists)
                    return BadRequest("FaceId does not exist.");

                // Save user details to UserDetails table
                var entity = new UserDetails
                {
                    FaceId = request.FaceId,
                    Name = request.Name,
                    PhoneNumber = request.PhoneNumber
                };

                _context.UserDetails.Add(entity);
                await _context.SaveChangesAsync();

                return Ok(new
                {
                    message = "User details stored successfully",
                    faceId = entity.FaceId,
                    name = entity.Name,
                    phoneNumber = entity.PhoneNumber
                });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { error = $"Storing user details failed: {ex.Message}" });
            }
        }


        [HttpPost("clear")]
        public async Task<IActionResult> ClearAllFaces()
        {
            try
            {
                _context.Faces.RemoveRange(_context.Faces);
                await _context.SaveChangesAsync();

                string dbPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "face_db");
                if (Directory.Exists(dbPath)) Directory.Delete(dbPath, true);

                string indexPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "face_index.faiss");
                if (System.IO.File.Exists(indexPath)) System.IO.File.Delete(indexPath);

                return Ok(new { message = "All enrolled faces and FAISS index cleared successfully." });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { error = $"Failed to clear data: {ex.Message}" });
            }
        }

    [HttpGet("ping")]
    public IActionResult Ping()
    {
      return Ok(new { message = "API is working good good" });
    }
  }
}
